from __future__ import print_function
import numpy as np

from legacypipe.decam import DecamImage
from legacypipe.bok import BokImage
from legacypipe.mosaic import MosaicImage
from legacypipe.survey import *
import tractor.ellipses as ellipses
import galsim
from astrometry.libkd.spherematch import match_radec
import logging
from tractor.sfd import SFDMap
from tractor.galaxy import DevGalaxy, ExpGalaxy
from tractor.sersic import SersicGalaxy
from legacypipe.survey import LegacySersicIndex
import tractor.ellipses as ellipses
import tractor
from tractor.basics import RaDecPos
from tractor import *
import subprocess

'''
Testing code for adding noise to deeper images to simulate DECaLS depth data.
'''

class SimDecals(LegacySurveyData):
    def __init__(self, dataset=None, survey_dir=None, metacat=None, simcat=None,
                 output_dir=None,add_sim_noise=False, seed=0, brickname=None,
                 **kwargs):
        self.dataset= dataset
        kw= dict(survey_dir=survey_dir,
                 output_dir=output_dir)
        #if self.dataset == 'cosmos':
        #    kw.update(subset=kwargs['subset'])
        super(SimDecals, self).__init__(**kw)

        self.metacat = metacat
        self.simcat = simcat
        # Additional options from command line
        self.add_sim_noise= add_sim_noise
        self.seed= seed
        self.brick = brickname
        self.no_sim=False
        if self.dataset == 'cosmos':
            self.cosmos_num = os.environ['cosmos_section']

    def get_image_object(self, t):
        if self.dataset == 'cosmos':
            return SimImageCosmos(self, t)
        else:
            if t.camera=='decam' or t.camera=='decam+noise':
                return SimImageDecam(self, t)
            elif t.camera=='90prime':
                return SimImageBok(self, t)
            elif t.camera=='mosaic':
                return SimImageMosaic(self, t)
            else:
                raise ValueError('no such camera!%s'%t.camera)
    def get_ccds(self, **kwargs):
        if self.dataset == 'cosmos':
            #cosmos hack: read from dustin's cosmos repeat ccd files
            print('CosmosSurvey.get_ccds()')
            import os
            brickname = os.environ['BRICKNAME']
            ccd_fn = '/global/cscratch1/sd/dstn/dr9-cosmos-subs/%s/coadd/%s/%s/legacysurvey-%s-ccds.fits'%(self.cosmos_num,brickname[:3],brickname,brickname)
            fns=[ccd_fn]

            TT = []
            for fn in fns:

                debug('Reading CCDs from', fn)
                T = fits_table(fn, **kwargs)
                debug('Got', len(T), 'CCDs')
                T.camera = np.array(['decam']*len(T),dtype=np.str)
                TT.append(T)
            if len(TT) > 1:
                T = merge_tables(TT, columns='fillzero')
            else:
                T = TT[0]
            debug('Total of', len(T), 'CCDs')
            del TT
            T = self.cleanup_ccds_table(T)
            return T
        else:
            CCDs = super(SimDecals, self).get_ccds(**kwargs)
            return CCDs
        
    def filter_ccd_kd_files(self, fns):
        """see legacypipe/runs.py"""
        return []


def noise_for_galaxy(gal,nano2e):
    """Returns numpy array of noise in Img count units for gal in image cnt units"""
    # Noise model + no negative image vals when compute noise
    one_std_per_pix= gal.array.copy() # nanomaggies
    one_std_per_pix[one_std_per_pix < 0]=0
    # rescale
    one_std_per_pix *= nano2e # e-
    one_std_per_pix= np.sqrt(one_std_per_pix)
    num_stds= np.random.randn(one_std_per_pix.shape[0],one_std_per_pix.shape[1])
    #one_std_per_pix.shape, num_stds.shape
    noise= one_std_per_pix * num_stds
    # rescale
    noise /= nano2e #nanomaggies
    return noise

def ivar_for_galaxy(gal,nano2e):
    """Adds gaussian noise to perfect source

    Args:
        gal: galsim.Image() for source, UNITS: nanomags
        nano2e: factor to convert to e- (gal * nano2e has units e-)

    Returns:
        galsim.Image() of invvar for the source, UNITS: nanomags
    """
    var= gal.copy() * nano2e #e^2
    var.applyNonlinearity(np.abs)
    var /= nano2e**2 #nanomag^2
    var.invertSelf()
    return var

def get_srcimg_invvar(stamp_ivar,img_ivar):
    """stamp_ivar, img_ivar -- galsim Image objects"""
    # Use img_ivar when stamp_ivar == 0, both otherwise
    use_img_ivar= np.ones(img_ivar.array.shape).astype(bool)
    use_img_ivar[ stamp_ivar.array > 0 ] = False
    # First compute using both
    ivar= np.power(stamp_ivar.array.copy(), -1) + np.power(img_ivar.array.copy(), -1)
    ivar= np.power(ivar,-1)
    keep= np.ones(ivar.shape).astype(bool)
    keep[ (stamp_ivar.array > 0)*\
          (img_ivar.array > 0) ] = False
    ivar[keep] = 0.
    # Now use img_ivar only where need to
    ivar[ use_img_ivar ] = img_ivar.array.copy()[ use_img_ivar ]
    # return
    obj_ivar = stamp_ivar.copy()
    obj_ivar.fill(0.)
    obj_ivar+= ivar
    return obj_ivar

class DecamImagePlusNoise(DecamImage):
    '''
    A DecamImage subclass to add noise to DECam images upon read.
    '''
    def __init__(self, survey, t):
        t.camera = 'decam'
        super(DecamImagePlusNoise, self).__init__(survey, t)
        self.addnoise = t.addnoise

    def get_tractor_image(self, **kwargs):
        
        assert(kwargs.get('nanomaggies', True))
        tim = super(DecamImagePlusNoise, self).get_tractor_image(**kwargs)
        if tim is None:
            return None
        with np.errstate(divide='ignore'):
            ie = 1. / (np.hypot(1. / tim.inverr, self.addnoise))
        ie[tim.inverr == 0] = 0.
        tim.inverr = ie
        tim.data += np.random.normal(size=tim.shape) * self.addnoise
        print('Adding noise: sig1 was', tim.sig1)
        print('Adding', self.addnoise)
        sig1 = 1. / np.median(ie[ie > 0])
        print('New sig1 is', sig1)
        tim.sig1 = sig1
        #tim.zr = [-3. * sig1, 10. * sig1]
        #tim.ima.update(vmin=tim.zr[0], vmax=tim.zr[1])
        return tim


class SimImage(object):
    def __init__(self, survey, t):
        super(SimImage, self).__init__(survey, t)
        self.t = t
        brickname = survey.brick
        brick = survey.get_brick_by_name(brickname)
        brickid = brick.brickid
        brickname = brick.brickname
        brickwcs = wcs_for_brick(brick)
        W, H, pixscale = brickwcs.get_width(), brickwcs.get_height(), brickwcs.pixel_scale()
        targetwcs = wcs_for_brick(brick, W=W, H=H, pixscale=pixscale)
        self.targetwcs = targetwcs

    def get_tractor_image(self, **kwargs):
        tim = super(SimImage, self).get_tractor_image(**kwargs)
        if tim is None: # this can be None when the edge of a CCD overlaps
            return tim
        #import pdb;pdb.set_trace()
        objstamp = BuildStamp(tim, targetwcs = self.targetwcs, camera = self.t.camera)

        tim.ids_added=[]

        tim_image = galsim.Image(tim.getImage())
        tim_invvar = galsim.Image(tim.getInvvar())
        tim_dq = galsim.Image(tim.dq)
        # Also store galaxy sims and sims invvar
        sims_image = tim_image.copy()
        sims_image.fill(0.0)
        sims_ivar = sims_image.copy()

        for ii, obj in enumerate(self.survey.simcat):
            strin= 'Drawing 1 galaxy: n=%.2f, rhalf=%.2f, e1=%.2f, e2=%.2f' % \
                        (obj.n,obj.rhalf,obj.e1,obj.e2)
            print(strin)
            stamp = objstamp.galaxy(obj)
            stamp_nonoise= stamp.copy()
            if self.survey.add_sim_noise:
                stamp += noise_for_galaxy(stamp,objstamp.nano2e)
            ivarstamp= ivar_for_galaxy(stamp,objstamp.nano2e)
            # Add source if EVEN 1 pix falls on the CCD
            overlap = stamp.bounds & tim_image.bounds
            if overlap.area() > 0:
                print('Stamp overlaps tim: id=%d band=%s' % (obj.id,objstamp.band))
                tim.ids_added.append(obj.id)
                stamp = stamp[overlap]
                ivarstamp = ivarstamp[overlap]
                stamp_nonoise= stamp_nonoise[overlap]

                # Zero out invvar where bad pixel mask is flagged (> 0)
                keep = np.ones(tim_dq[overlap].array.shape)
                keep[ tim_dq[overlap].array > 0 ] = 0.
                ivarstamp *= keep
                # Add stamp to image
                back= tim_image[overlap].copy()
                tim_image[overlap] += stamp #= back.copy() + stamp.copy()
                # Add variances
                back_ivar= tim_invvar[overlap].copy()
                tot_ivar= get_srcimg_invvar(ivarstamp, back_ivar)
                tim_invvar[overlap] = tot_ivar.copy()

                #Extra
                sims_image[overlap] += stamp.copy()
                sims_ivar[overlap] += ivarstamp.copy()

        tim.sims_image = sims_image.array
        tim.sims_inverr = np.sqrt(sims_ivar.array)
        # Can set image=model, ivar=1/model for testing
        tim.data = tim_image.array
        tim.inverr = np.sqrt(tim_invvar.array)
        sys.stdout.flush()
        #print('get_tractor_image in obiwan :time '+str(time_builtin.clock()-t1))
        return tim


class SimImageDecam(SimImage, DecamImage):
    def __init__(self, survey, t):
        super(SimImageDecam, self).__init__(survey, t)

class SimImageBok(SimImage, BokImage):
      def __init__(self, survey, t):
        super(SimImageBok, self).__init__(survey, t)
class SimImageMosaic(SimImage, MosaicImage):
      def __init__(self, survey, t):
        super(SimImageMosaic, self).__init__(survey, t)

class SimImageCosmos(SimImage,DecamImagePlusNoise):
    def __init__(self, survey, t):
        super(SimImageCosmos, self).__init__(survey, t)



class BuildStamp():
    def __init__(self, tim, targetwcs=None, camera=None):
        if camera == 'decam+noise':
                camera = 'decam'
        assert(camera in ['decam','mosaic','90prime'])

        if camera == 'decam':
               self.nano2e = tim.zpscale*tim.gain
        else:
                self.nano2e = tim.zpscale*tim.exptime
        self.targetwcs = targetwcs
        self.tim = tim
        self.band = tim.band

    # local image that the src resides in 
    def set_local(self, obj):
        ra, dec = obj.ra, obj.dec
        flag, target_x, target_y = self.targetwcs.radec2pixelxy(ra, dec)
        #intentionally setting source center in the center pixel on the coadd image
        #target_x, target_y = int(target_x+0.5), int(target_y+0.5)
        #import pdb;pdb.set_trace()
        #ra_new, dec_new = self.targetwcs.pixelxy2radec(target_x, target_y)[-2:]
        
        x=int(obj.get('x')+0.5)
        y=int(obj.get('y')+0.5)
        self.target_x=x
        self.target_y=y
        
        self.ra, self.dec = ra, dec
        flag, xx, yy = self.tim.subwcs.radec2pixelxy(*(self.targetwcs.pixelxy2radec(self.target_x+1, self.target_y+1)[-2:]))
        x_cen = xx-1
        y_cen = yy-1
        self.wcs=self.tim.getWcs() 
        x_cen_int,y_cen_int = round(x_cen),round(y_cen)
        self.sx0,self.sx1,self.sy0,self.sy1 = x_cen_int-32,x_cen_int+31,y_cen_int-32,y_cen_int+31
        (h,w) = self.tim.shape
        self.sx0 = np.clip(int(self.sx0), 0, w-1)
        self.sx1 = np.clip(int(self.sx1), 0, w-1) + 1
        self.sy0 = np.clip(int(self.sy0), 0, h-1)
        self.sy1 = np.clip(int(self.sy1), 0, h-1) + 1
        subslc = slice(self.sy0,self.sy1),slice(self.sx0,self.sx1)
        subimg = self.tim.getImage ()[subslc]
        subie  = self.tim.getInvError()[subslc]
        subwcs = self.tim.getWcs().shifted(self.sx0, self.sy0)
        subsky = self.tim.getSky().shifted(self.sx0, self.sy0)
        subpsf = self.tim.psf.constantPsfAt((self.sx0+self.sx1)/2., (self.sy0+self.sy1)/2.)
        new_tim = tractor.Image(data=subimg, inverr=subie, wcs=subwcs,psf=subpsf, photocal=self.tim.getPhotoCal(), sky=subsky, name=self.tim.name)
        return new_tim
    def galaxy(self, obj):
        new_tim = self.set_local(obj)
        n,r_half,e1,e2,flux = int(obj.get('n')),float(obj.get('rhalf')),float(obj.get('e1')),float(obj.get('e2')),float(obj.get(self.band+'flux'))
        assert(self.band in ['g','r','z'])
        if self.band == 'g':
               brightness =  tractor.NanoMaggies(g=flux, order=['g'])
        if self.band == 'r':
                brightness =  tractor.NanoMaggies(r=flux, order=['r'])
        if self.band == 'z':
                brightness =  tractor.NanoMaggies(z=flux, order=['z'])
        shape = ellipses.EllipseE(r_half,e1,e2)
        if n==1:
              new_gal = ExpGalaxy(RaDecPos(self.ra, self.dec), brightness, shape)
        elif n==4:
              new_gal = DevGalaxy(RaDecPos(self.ra, self.dec), brightness, shape)
        elif n==0:
              new_gal = PointSource(RaDecPos(self.ra, self.dec), brightness)
        else:
              new_gal = SersicGalaxy(RaDecPos(self.ra, self.dec), brightness, shape, LegacySersicIndex(n))

        new_tractor = Tractor([new_tim], [new_gal])
        mod0 = new_tractor.getModelImage(0)
        galsim_img = galsim.Image(mod0)
        galsim_img.bounds.xmin=self.sx0+1
        galsim_img.bounds.xmax=self.sx1-1
        galsim_img.bounds.ymin=self.sy0+1
        galsim_img.bounds.ymax=self.sy1-1
        return galsim_img




def get_parser_sim():
    from legacypipe.runbrick import get_parser
    parser = get_parser()
    parser.add_argument('--nobj', type=int, help='total number of objects to be injected', default=0)
    parser.add_argument('--startid', type=int, help='starting index of randoms to be injected', default=0)
    parser.add_argument('--random_fn', default=None, help='directory of input randoms')
    parser.add_argument('--add_sim_noise',  action="store_true", help="set to add noise to simulated sources")
    parser.add_argument('--dataset', default='normal', help='choose from Normal, cosmos')
    parser.add_argument('--subset', type=int, help='COSMOS repeats subset number, 80-89', default=0)
    parser.add_argument('--all-blobs', action='store_true',default=False,
                        help='fit models to all blobs, not just those containing sim sources')
    parser.add_argument('--no_sim',action='store_true',default=False,help='run the whole code without obiwan, in principle the same as running legacypipe')
    return parser

def flag_nearest_neighbors(Samp, radius_in_deg=5./3600):
  """Returns Sample indices to keep (have > dist separations) and indices to skip

  Returns:
    tuple: keep,skip: indices of Samp to keep and skip
  """
  flag_set=set()
  all_indices= range(len(Samp))
  for cnt in all_indices:
      if cnt in flag_set:
          continue
      else:
          I,J,d = match_radec(Samp.ra[cnt],Samp.dec[cnt],
                              Samp.ra,Samp.dec, 5./3600,
                              notself=False,nearest=False)
          # Remove all Samp matches (J), minus the reference ra,dec
          flag_inds= set(J).difference(set( [cnt] ))
          if len(flag_inds) > 0:
              flag_set= flag_set.union(flag_inds)
  keep= list( set(all_indices).difference(flag_set) )
  return keep, list(flag_set)

def build_simcat(Samp=None, brickname = None, radius_in_deg=5./3600):
    log = logging.getLogger('decals_sim')

    i_keep,i_skip= flag_nearest_neighbors(Samp, radius_in_deg=radius_in_deg)
    skipping_ids= Samp.get('id')[i_skip]
    log.info('sources %d, keeping %d, flagged as nearby %d' % (len(Samp),len(i_keep),len(i_skip)))
    Samp.cut(i_keep)

    survey = LegacySurveyData(survey_dir=None)
    brickinfo= survey.get_brick_by_name(brickname)
    brickwcs = wcs_for_brick(brickinfo)
    W, H, pixscale = brickwcs.get_width(), brickwcs.get_height(), brickwcs.pixel_scale()
    targetwcs = wcs_for_brick(brickinfo, W=W, H=H, pixscale=pixscale)
    flag, xx,yy = targetwcs.radec2pixelxy(Samp.ra,Samp.dec)

    cat = fits_table()
    for key in ['id','ra','dec']:
        cat.set(key, Samp.get(key))
    cat.set('x', xx-1)
    cat.set('y', yy-1)

    # Mags
    filts = ['%s %s' % ('DES', f) for f in 'grz']
    filts.append('WISE W1')
    filts.append('WISE W2')
    ii = 0
    for band in ['g','r','z','w1','w2']:
        nanomag= 1E9*10**(-0.4*Samp.get(band))
        # Add extinction (to stars too, b/c "decam-chatter 6517")
        mw_transmission= SFDMap().extinction([filts[ii]],
                                             Samp.ra, Samp.dec)
        mw_transmission= 10**(-mw_transmission[:,0].astype(np.float32)/2.5)
        cat.set('%sflux' % band, nanomag * mw_transmission)
        cat.set('mw_transmission_%s' % band, mw_transmission)
        ii+=1

    for key in ['n','rhalf','e1','e2']:
            cat.set(key, Samp.get(key))

    return cat, skipping_ids

def main():
    from legacypipe.runbrick import run_brick, get_runbrick_kwargs

    parser = get_parser_sim()
    # subset number
    
    opt = parser.parse_args()
    if opt.brick is None and opt.radec is None:
        parser.print_help()
        return -1
    
    
    optdict = vars(opt)
    verbose = optdict.pop('verbose')
    subset = optdict.pop('subset')
    brickname = opt.brick
    nobj = optdict.pop('nobj')
    startid = optdict.pop('startid')
    random_fn = optdict.pop('random_fn')
    add_sim_noise = optdict.pop('add_sim_noise')
    all_blobs = optdict.pop('all_blobs')
    dataset = optdict.pop('dataset')
    no_sim = optdict.pop('no_sim')

    samp = fits_table(random_fn)[startid:startid+nobj]
    simcat, skipped = build_simcat(samp,brickname)
    import os
    simcat_dir = os.path.join(opt.output_dir,'coadd',brickname[:3],brickname)
    subprocess.call(["mkdir","-p",simcat_dir])
    simcat.writeto(os.path.join(opt.output_dir,'coadd',brickname[:3],brickname,'legacysurvey-simcat-%s.fits'%brickname))
    import logging
    
    #tractor_fn = opt.output_dir+'/tractor/%s/tractor-%s.fits'%(brickname[:3],brickname)
    #if os.path.isfile(tractor_fn):
    #    print("brick %s already finished"%brickname)
    #    return None
    
    if verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    if no_sim:
        survey, kwargs = get_runbrick_kwargs(**optdict)
        survey.no_sim = True
        survey.simcat = None
        if kwargs in [-1,0]:
            return kwargs
        print('Using survey:', survey)
        run_brick(opt.brick, survey, **kwargs)
        return 0

    if all_blobs:
        blobxy = None
    else:
        blobxy = zip(simcat.get('x'), simcat.get('y'))

    optdict.update(blobxy=blobxy)


    survey, kwargs = get_runbrick_kwargs(**optdict)
    if kwargs in [-1,0]:
        return kwargs
    

    survey = SimDecals(survey_dir=opt.survey_dir, subset=subset, simcat = simcat, dataset=dataset,
                          output_dir=opt.output_dir,brickname=brickname, add_sim_noise=add_sim_noise)
    print('Using survey:', survey)
    #print('with output', survey.output_dir)

    print('opt:', opt)
    print('kwargs:', kwargs)

    run_brick(opt.brick, survey, **kwargs)
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())

