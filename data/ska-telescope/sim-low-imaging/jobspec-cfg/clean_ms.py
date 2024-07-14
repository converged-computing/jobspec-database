import argparse
import logging
import os
import pprint
import sys
import time

import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt

import numpy
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u

from rascil.data_models import ReceptorFrame, PolarisationFrame
from rascil.processing_components import create_awterm_convolutionfunction, qa_image, export_image_to_fits, show_image, \
    import_image_from_fits, advise_wide_field, create_image_from_visibility, create_blockvisibility_from_ms, \
    vis_summary, convert_blockvisibility_to_visibility, coalesce_visibility, convert_convolutionfunction_to_image, \
    convert_blockvisibility_to_stokesI, create_calibration_controls

from rascil.workflows import weight_list_rsexecute_workflow, invert_list_rsexecute_workflow, \
    sum_invert_results_rsexecute, continuum_imaging_list_rsexecute_workflow, ical_list_rsexecute_workflow

from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute

pp = pprint.PrettyPrinter()
cwd = os.getcwd()


def init_logging():
    log.info("Logging to %s/clean_ms_dask.log" % cwd)
    logging.basicConfig(filename='%s/clean_ms_dask.log' % cwd,
                        filemode='a',
                        format='%(thread)s %(asctime)s.%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)


if __name__ == "__main__":
    
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler(sys.stdout))
    log.addHandler(logging.StreamHandler(sys.stderr))
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.WARNING)
    init_logging()
    
    start_epoch = time.asctime()
    log.info("\nSKA LOW imaging using RASCIL\nStarted at %s\n" % start_epoch)
    
    ########################################################################################################################
    
    parser = argparse.ArgumentParser(description='SKA LOW imaging using RASCIL')
    parser.add_argument('--context', type=str, default='2d', help="Imaging context "'2d'|'ng')
    parser.add_argument('--mode', type=str, default='pipeline', help="Imaging mode 'invert'|'cip'|'ical'")
    parser.add_argument('--msname', type=str, default='../data/EoR0_20deg_24.MS',
                        help='MS to process')
    parser.add_argument('--model_image', type=str, default=None, help='Initial model image')
    parser.add_argument('--local_directory', type=str, default='dask-workspace',
                        help='Local directory for Dask files')
    
    parser.add_argument('--channels', type=int, nargs=2, default=[0, 160], help='Channels to process')
    parser.add_argument('--ngroup', type=int, default=4,
                        help='Number of channels in each BlockVisibility')
    parser.add_argument('--single', type=str, default='False', help='Use a single channel')
    
    parser.add_argument('--time_coal', type=float, default=0.0, help='Coalesce time')
    parser.add_argument('--frequency_coal', type=float, default=0.0, help='Coalesce frequency')
    
    parser.add_argument('--npixel', type=int, default=None, help='Number of pixels')
    parser.add_argument('--fov', type=float, default=1.0, help='Field of view in primary beams')
    parser.add_argument('--cellsize', type=float, default=None, help='Cellsize in radians')
    
    parser.add_argument('--wstep', type=float, default=None, help='Step in w (wavelengths)')
    parser.add_argument('--nwplanes', type=int, default=None, help='Number of w planes')
    parser.add_argument('--nwslabs', type=int, default=None, help='Number of w slabs')
    parser.add_argument('--amplitude_loss', type=float, default=0.02, help='Amplitude loss due to w sampling')
    parser.add_argument('--oversampling', type=int, default=16, help='Oversampling in w projection kernel')

    parser.add_argument('--epsilon', type=float, default=1e-12, help='Accuracy in nifty gridder')
    
    parser.add_argument('--weighting', type=str, default='natural', help='Type of weighting')
    
    parser.add_argument('--nmoment', type=int, default=1, help='Number of spectral moments in MSMFS Clean')
    parser.add_argument('--nmajor', type=int, default=1, help='Number of major cycles')
    parser.add_argument('--niter', type=int, default=1000, help='Number of iterations per major cycle')
    parser.add_argument('--fractional_threshold', type=float, default=0.2,
                        help='Fractional threshold to terminate major cycle')
    parser.add_argument('--threshold', type=float, default=0.01, help='Absolute threshold (Jy) to terminate')
    parser.add_argument('--window_shape', type=str, default=None, help="Window shape 'quarter'|'no_edge")
    parser.add_argument('--window_edge', type=int, default=None, help='Window edge (pixels)')
    parser.add_argument('--deconvolve_facets', type=int, default=1, help='Number of facets in deconvolution')
    parser.add_argument('--deconvolve_overlap', type=int, default=128, help='Overlap in deconvolution')
    parser.add_argument('--deconvolve_taper', type=str, default='tukey', help="Overlap function 'linear'|'tukey'")
    parser.add_argument('--restore_facets', type=int, default=1, help='Number of facets in restore')

    parser.add_argument('--serial', type=str, default='False', help='Use serial processing throughout?')
    parser.add_argument('--nworkers', type=int, default=4, help='Number of workers for Dask processing')
    parser.add_argument('--threads_per_worker', type=int, default=1, help='Number of threads per Dask worker')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads for nifty gridder')
    parser.add_argument('--memory', type=int, default=64, help='Memory of each worker (GB)')
    
    parser.add_argument('--use_serial_invert', type=str, default='False', help='Use serial invert?')
    parser.add_argument('--use_serial_predict', type=str, default='False', help='Use serial predict?')
    parser.add_argument('--plot', type=str, default='False', help='Plot data?')
    
    args = parser.parse_args()
    
    pp.pprint(vars(args))
    
    target_ms = args.msname
    log.info("Target MS is %s" % target_ms)
    msext = '.' + (target_ms.split('.')[-1])
    log.info("MS extension is %s" % msext)
    
    initial_model = args.model_image
    
    ochannels = numpy.arange(args.channels[0], args.channels[1] + 1)
    nmoment = args.nmoment
    ngroup = args.ngroup
    weighting = args.weighting
    nwplanes = args.nwplanes
    nwslabs = args.nwslabs
    npixel = args.npixel
    cellsize = args.cellsize
    mode = args.mode
    fov = args.fov
    wstep = args.wstep
    context = args.context
    use_serial_invert = args.use_serial_invert == "True"
    use_serial_predict = args.use_serial_predict == "True"
    serial = args.serial == "True"
    plot = args.plot == "True"
    single = args.single == "True"
    nworkers = args.nworkers
    threads_per_worker = args.threads_per_worker
    memory = args.memory
    time_coal = args.time_coal
    frequency_coal = args.frequency_coal
    local_directory = args.local_directory
    window_edge = args.window_edge
    window_shape = args.window_shape
    dela = args.amplitude_loss
    
    if context == 'ng':
        use_block = True
    else:
        use_block = False
    
    ####################################################################################################################
    
    log.info("\nSetup of processing mode")
    if serial:
        log.info("Will use serial processing")
        use_serial_invert = True
        use_serial_predict = True
        rsexecute.set_client(use_dask=False)
        print(rsexecute.client)
    else:
        
        from dask.distributed import Client
        
        scheduler = os.getenv('RASCIL_DASK_SCHEDULER', None)
        if scheduler is not None:
            log.info("Creating Dask Client using externally defined scheduler")
            client = Client(scheduler)
        else:
            log.info("Using Dask on this computer")
            client = Client(threads_per_worker=threads_per_worker, n_workers=nworkers,
                            memory_limit=128 * 1024 * 1024 * 1024)
        
        rsexecute.set_client(client=client)
        
        print(rsexecute.client)
        if use_serial_invert:
            log.info("Will use serial invert")
        else:
            log.info("Will use distributed invert")
        if use_serial_predict:
            log.info("Will use serial predict")
        else:
            log.info("Will use distributed predict")
        
        rsexecute.client.run(init_logging)
    
    ####################################################################################################################
    
    # Read an MS and convert to Visibility format
    log.info("\nSetup of visibility ingest")
    
    
    def read_convert(ms, ch):
        start = time.time()
        bvis = create_blockvisibility_from_ms(ms, start_chan=ch[0], end_chan=ch[1])[0]
        # The following are not set in the MSes
        bvis.configuration.location = EarthLocation(lon="116.76444824", lat="-26.824722084", height=300.0)
        bvis.configuration.frame = ""
        bvis.configuration.receptor_frame = ReceptorFrame("linear")
        bvis.configuration.data['diameter'][...] = 35.0
        
        print(bvis)
        if bvis.polarisation_frame != PolarisationFrame('stokesI'):
            log.info("Converting visibility to stokesI")
            bvis = convert_blockvisibility_to_stokesI(bvis)
        print(bvis)
        
        if use_block:
            log.info("Working with BlockVisibility's")
            return bvis
        else:
            if time_coal > 0.0 or frequency_coal > 0.0:
                vis = coalesce_visibility(bvis, time_coal=time_coal, frequency_coal=frequency_coal)
                log.info(
                    "Time to read and convert %s, channels %d to %d = %.1f s" % (ms, ch[0], ch[1], time.time() - start))
                print('Size of visibility before compression %s, after %s' % (vis_summary(bvis), vis_summary(vis)))
            else:
                vis = convert_blockvisibility_to_visibility(bvis)
                log.info(
                    "Time to read and convert %s, channels %d to %d = %.1f s" % (ms, ch[0], ch[1], time.time() - start))
                print('Size of visibility before conversion %s, after %s' % (vis_summary(bvis), vis_summary(vis)))
            del bvis
            return vis
    
    
    channels = []
    for i in range(0, len(ochannels) - 1, ngroup):
        channels.append([ochannels[i], ochannels[i + ngroup - 1]])
    print(channels)
    
    if single:
        channels = [channels[0]]
        log.info("Will read single range of channels %s" % channels)
    
    vis_list = [rsexecute.execute(read_convert)(target_ms, group_chan) for group_chan in channels]
    vis_list = rsexecute.persist(vis_list)
    
    ####################################################################################################################
    
    log.info("\nSetup of images")
    phasecentre = SkyCoord(ra=0.0 * u.deg, dec=-27.0 * u.deg)
    
    advice = [rsexecute.execute(advise_wide_field)(v, guard_band_image=fov, delA=dela, verbose=(iv == 0))
              for iv, v in enumerate(vis_list)]
    advice = rsexecute.compute(advice, sync=True)
    
    if npixel is None:
        npixel = advice[0]['npixels_min']
    
    if wstep is None:
        wstep = 1.1 * advice[0]['wstep']
    
    if nwplanes is None:
        nwplanes = advice[0]['wprojection_planes']
    
    if cellsize is None:
        cellsize = advice[-1]['cellsize']
    
    log.info('Image shape is %d by %d pixels' % (npixel, npixel))
    
    ####################################################################################################################
    
    log.info("\nSetup of wide field imaging")
    vis_slices = 1
    actual_context = '2d'
    support = 1
    if context == 'wprojection':
        # w projection
        vis_slices = 1
        support = advice[0]['nwpixels']
        actual_context = '2d'
        log.info("Will do w projection, %d planes, support %d, step %.1f" %
                 (nwplanes, support, wstep))
    
    
    elif context == 'ng':
        # nifty gridder
        log.info("Will use nifty gridder")
        actual_context = 'ng'
    
    elif context == 'wstack':
        # w stacking
        log.info("Will do w stack, %d planes, step %.1f" % (nwplanes, wstep))
        actual_context = 'wstack'
    
    elif context == 'wprojectwstack':
        # Hybrid w projection/wstack
        nwplanes = int(1.5 * nwplanes) // nwslabs
        support = int(1.5 * advice[0]['nwpixels'] / nwslabs)
        support = max(15, int(3.0 * advice[0]['nwpixels'] / nwslabs))
        support -= support % 2
        vis_slices = nwslabs
        actual_context = 'wstack'
        log.info("Will do hybrid w stack/w projection, %d w slabs, %d w planes, support %d, w step %.1f" %
                 (nwslabs, nwplanes, support, wstep))
    else:
        log.info("Will do 2d processing")
        # Simple 2D
        actual_context = '2d'
        vis_slices = 1
        wstep = 1e15
        nwplanes = 1
    
    if initial_model is None:
        model_list = [rsexecute.execute(create_image_from_visibility)(v, npixel=npixel, cellsize=cellsize)
                      for v in vis_list]
    else:
        model_list = [rsexecute.execute(import_image_from_fits)(initial_model)
                      for v in vis_list]
    
    model = rsexecute.compute(model_list[0], sync=True)
    
    # Perform weighting. This is a collective computation, requiring all visibilities :(
    log.info("\nSetup of weighting")
    if weighting == 'uniform':
        log.info("Will apply uniform weighting")
        vis_list = weight_list_rsexecute_workflow(vis_list, model_list)
    
    if context == 'wprojection' or context == 'wprojectwstack':
        gcfcf_list = [rsexecute.execute(create_awterm_convolutionfunction)(m, nw=nwplanes, wstep=wstep,
                                                                            oversampling=args.oversampling,
                                                                            support=support,
                                                                            maxsupport=512)
                      for m in model_list]
        gcfcf_list = rsexecute.persist(gcfcf_list)
        gcfcf = rsexecute.compute(gcfcf_list[0], sync=True)
        cf = convert_convolutionfunction_to_image(gcfcf[1])
        cf.data = numpy.real(cf.data)
        export_image_to_fits(cf, "cf.fits")
    else:
        gcfcf_list = None
    
    ####################################################################################################################
    
    rsexecute.init_statistics()
    
    if mode == 'pipeline':
        log.info("\nRunning pipeline")
        cip_result = continuum_imaging_list_rsexecute_workflow(vis_list, model_list, context=actual_context,
                                                                vis_slices=vis_slices,
                                                                use_serial_invert=use_serial_invert,
                                                                use_serial_predict=use_serial_predict,
                                                                niter=args.niter,
                                                                fractional_threshold=args.fractional_threshold,
                                                                threshold=args.threshold,
                                                                nmajor=args.nmajor, gain=0.1,
                                                                algorithm='mmclean',
                                                                nmoment=nmoment, findpeak='RASCIL',
                                                                scales=[0],
                                                                restore_facets=args.restore_facets,
                                                                psfwidth=1.0,
                                                                deconvolve_facets=args.deconvolve_facets,
                                                                deconvolve_overlap=args.deconvolve_overlap,
                                                                deconvolve_taper=args.deconvolve_taper,
                                                                timeslice='auto',
                                                                psf_support=256,
                                                                window_shape=window_shape,
                                                                window_edge=window_edge,
                                                                gcfcf=gcfcf_list,
                                                                return_moments=False,
                                                                threads=args.threads,
                                                                epsilon=args.epsilon)
        
        start = time.time()
        result = rsexecute.compute(cip_result, sync=True)
        run_time = time.time() - start
        log.info("Processing took %.2f (s)" % run_time)
        
        pp.pprint(result)
        deconvolved = result[0][0]
        residual = result[1][0][0]
        restored = result[2][0]
        
        print(qa_image(restored))
        
        restored_name = target_ms.split('/')[-1].replace(msext, '_cip_restored.fits')
        log.info("Writing restored image to %s" % restored_name)
        export_image_to_fits(restored, restored_name)
        
        title = target_ms.split('/')[-1].replace(msext, ' cip restored image')
        show_image(restored, vmax=0.03, vmin=-0.003, title=title)
        plot_name = target_ms.split('/')[-1].replace(msext, '_cip_restored.jpg')
        plt.savefig(plot_name)
        plt.show(block=False)
        
        deconvolved_name = target_ms.split('/')[-1].replace(msext, '_cip_deconvolved.fits')
        log.info("Writing cip deconvolved image to %s" % deconvolved_name)
        export_image_to_fits(deconvolved, deconvolved_name)
        
        residual_name = target_ms.split('/')[-1].replace(msext, '_cip_residual.fits')
        log.info("Writing residual image to %s" % residual_name)
        export_image_to_fits(residual, residual_name)
    
    elif mode == 'ical':
        
        controls = create_calibration_controls()
        
        controls['T']['first_selfcal'] = 0
        
        controls['T']['timeslice'] = 60.0
        controls['T']['phase_only'] = True
        
        log.info("\nRunning ical pipeline")
        ical_result = ical_list_rsexecute_workflow(vis_list, model_list, context=actual_context,
                                                    vis_slices=vis_slices,
                                                    use_serial_invert=use_serial_invert,
                                                    use_serial_predict=use_serial_predict,
                                                    niter=args.niter,
                                                    fractional_threshold=args.fractional_threshold,
                                                    threshold=args.threshold,
                                                    nmajor=args.nmajor, gain=0.1,
                                                    algorithm='mmclean',
                                                    nmoment=nmoment, findpeak='RASCIL',
                                                    scales=[0],
                                                    restore_facets=args.restore_facets,
                                                    psfwidth=1.0,
                                                    deconvolve_facets=args.deconvolve_facets,
                                                    deconvolve_overlap=args.deconvolve_overlap,
                                                    deconvolve_taper=args.deconvolve_taper,
                                                    timeslice='auto',
                                                    psf_support=256,
                                                    window_shape=window_shape,
                                                    window_edge=window_edge,
                                                    gcfcf=gcfcf_list,
                                                    return_moments=False,
                                                    calibration_context='T',
                                                    controls=controls,
                                                    threads=args.threads,
                                                    epsilon=args.epsilon)
        
        start = time.time()
        result = rsexecute.compute(ical_result, sync=True)
        run_time = time.time() - start
        log.info("Processing took %.2f (s)" % run_time)
        
        pp.pprint(result)
        deconvolved = result[0][0]
        residual = result[1][0][0]
        restored = result[2][0]
        
        print(qa_image(restored))
        
        restored_name = target_ms.split('/')[-1].replace(msext, '_ical_restored.fits')
        log.info("Writing restored image to %s" % restored_name)
        export_image_to_fits(restored, restored_name)
        
        title = target_ms.split('/')[-1].replace(msext, ' ical restored image')
        show_image(restored, vmax=0.03, vmin=-0.003, title=title)
        plot_name = target_ms.split('/')[-1].replace(msext, '_ical_restored.jpg')
        plt.savefig(plot_name)
        plt.show(block=False)
        
        deconvolved_name = target_ms.split('/')[-1].replace(msext, '_ical_deconvolved.fits')
        log.info("Writing deconvolved image to %s" % deconvolved_name)
        export_image_to_fits(deconvolved, deconvolved_name)
        
        residual_name = target_ms.split('/')[-1].replace(msext, '_ical_residual.fits')
        log.info("Writing residual image to %s" % residual_name)
        export_image_to_fits(residual, residual_name)
    
    else:
        log.info("\nRunning invert")
        result = invert_list_rsexecute_workflow(vis_list, model_list, context=actual_context, vis_slices=nwplanes,
                                                 use_serial_invert=use_serial_invert,
                                                 gcfcf=gcfcf_list,
                                                 threads=args.threads,
                                                 epsilon=args.epsilon)
        result = sum_invert_results_rsexecute(result)
        result = rsexecute.persist(result)
        dirty = result[0]
        
        start = time.time()
        dirty = rsexecute.compute(dirty, sync=True)
        run_time = time.time() - start
        log.info("Processing took %.2f (s)" % run_time)
        
        print(qa_image(dirty))
        
        title = target_ms.split('/')[-1].replace(msext, ' dirty image')
        show_image(dirty, vmax=0.03, vmin=-0.003, title=title)
        plot_name = target_ms.split('/')[-1].replace(msext, '_dirty.png')
        plt.savefig(plot_name)
        plt.show(block=False)
        
        dirty_name = target_ms.split('/')[-1].replace(msext, '_dirty.fits')
        log.info("Writing dirty image to %s" % dirty_name)
        export_image_to_fits(dirty, dirty_name)
    
    rsexecute.save_statistics(name='clean_ms')
    
    if not serial:
        rsexecute.close()
    
    log.info("\nSKA LOW imaging using RASCIL")
    log.info("Started at  %s" % start_epoch)
    log.info("Finished at %s" % time.asctime())
