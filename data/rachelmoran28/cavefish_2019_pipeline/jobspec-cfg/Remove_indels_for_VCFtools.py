import os, sys, re
import subprocess
from collections import OrderedDict
from operator import itemgetter
from itertools import groupby

#to run
#python Remove_indels_and_masking_regions_bb.py /Users/suzannemcgaugh/Documents/University_of_Minnesota/CaveFish/Population_genomics/PASS_SNP_invariant_INDELTEST2.vcf /Users/suzannemcgaugh/Documents/University_of_Minnesota/CaveFish/Population_genomics/Repeat_masker_coordinates.txt>out4ff.txt
#python Remove_indels_and_masking_regions_for_VCFtools.py /panfs/roc/groups/14/mcgaughs/mcgaughs/projects/Astyanax/VCFs/filtered/Astyanax_mexicanus_pops_3.3.0_PASS.vcf /panfs/roc/scratch/mcgaugh/VCFs/Repeat_masker_coordinates.txt>exclude_VCFtools.txt

#next step
#cd /panfs/roc/groups/14/mcgaughs/mcgaughs/tools/vcftools_0.1.12b/cpp

#or on desktop

#cd  /Users/suzannemcgaugh/Desktop/vcftools/cpp
#./vcftools --vcf /Users/suzannemcgaugh/Documents/University_of_Minnesota/CaveFish/Population_genomics/PASS_SNP_invariant_INDELTEST2.vcf --remove /Users/suzannemcgaugh/Documents/University_of_Minnesota/CaveFish/Population_genomics/No_Tinaja.txt --site-pi --keep /Users/suzannemcgaugh/Documents/University_of_Minnesota/CaveFish/Population_genomics/Population_Tinaja.txt  --max-missing-count 8  --exclude-bed /Users/suzannemcgaugh/Documents/University_of_Minnesota/CaveFish/Population_genomics/Repeat_masker_coordinates.txt  --out /Users/suzannemcgaugh/Documents/University_of_Minnesota/CaveFish/Population_genomics/output_Pi_Tinaja10.txt


#./vcftools --vcf /Users/suzannemcgaugh/Documents/University_of_Minnesota/CaveFish/Population_genomics/Astyanax_mexicanus_pops_3.3.0_PASS_TEST.vcf --remove /Users/suzannemcgaugh/Documents/University_of_Minnesota/CaveFish/Population_genomics/No_Tinaja.txt --site-pi --keep /Users/suzannemcgaugh/Documents/University_of_Minnesota/CaveFish/Population_genomics/Population_Tinaja.txt  --max-missing-count 8  --exclude-bed /Users/suzannemcgaugh/Documents/University_of_Minnesota/CaveFish/Population_genomics/exclude.txt  --out /Users/suzannemcgaugh/Documents/University_of_Minnesota/CaveFish/Population_genomics/output_Pi_Tinaja10.txt

#/panfs/roc/scratch/mcgaugh/VCFs/Astyanax_mexicanus_pops_3.3.0_PASS_TEST.vcf

#/panfs/roc/groups/14/mcgaughs/mcgaughs/projects/Astyanax/VCFs/filtered/Astyanax_mexicanus_pops_3.3.0_PASS_TEST.vcf

hashy3={}
      
for line in open(sys.argv[1]):    
    line = line.strip().split('\t')
    line3=[]
    line7=[]
    if line[0][0:2]!='NW':
        len(line)
    elif line[0][0:2]=='NW': # this says only count lines starting with 'NW'. Must be changed if scaffolds have different start names
        line3=line[3].split(",")
        line7=line[7].split(",")
        Aeneus_granadas_cave=line[9].split(":")
        Aeneus_surface=line[10].split(":")
        Arroyo_1Boro=line[11].split(":")
        Arroyo_2_S12=line[12].split(":")
        Arroyo_3_S13=line[13].split(":")
        Arroyo_A2POG=line[14].split(":")
        Arroyo_B2POG=line[15].split(":")
        Arroyo_C4POG=line[16].split(":")
        Arroyo_D4POG=line[17].split(":")
        Arroyo_E4POG=line[18].split(":")
        Arroyo_F4POG=line[19].split(":")
        Arroyo_G4POG=line[20].split(":")
        CabMoro_C1Rnd1=line[21].split(":")
        CabMoro_C2Rnd1=line[22].split(":")
        CabMoro_C3Rnd1=line[23].split(":")
        CabMoro_C3Rnd2=line[24].split(":")
        CabMoro_C4Rnd1=line[25].split(":")
        CabMoro_C4Rnd2=line[26].split(":")
        CabMoro_C5Rnd1=line[27].split(":")
        CabMoro_C5Rnd2=line[28].split(":")
        CabMoro_C6Rnd1=line[29].split(":")
        CabMoro_C6Rnd2=line[30].split(":")
        CabMoro_C7Rnd1=line[31].split(":")
        CabMoro_C7Rnd2=line[32].split(":")
        CabMoro_E10Rnd2=line[33].split(":")
        CabMoro_E11Rnd2=line[34].split(":")
        CabMoro_E12Rnd2=line[35].split(":")
        CabMoro_E13Rnd2=line[36].split(":")
        CabMoro_E14Rnd2=line[37].split(":")
        CabMoro_E15Rnd2=line[38].split(":")
        CabMoro_E16Rnd2=line[39].split(":")
        CabMoro_E1Rnd1=line[40].split(":")
        CabMoro_E2Rnd1=line[41].split(":")
        CabMoro_E3Rnd1=line[42].split(":")
        CabMoro_E4Rnd1=line[43].split(":")
        CabMoro_E5Rnd1=line[44].split(":")
        CabMoro_E6Rnd1=line[45].split(":")
        CabMoro_E7Rnd1=line[46].split(":")
        CabMoro_E8Rnd1=line[47].split(":")
        CabMoro_E8Rnd2=line[48].split(":")
        CabMoro_E9Rnd2=line[49].split(":")
        CabMoro_S1Rnd1=line[50].split(":")
        CabMoro_S2Rnd1=line[51].split(":")
        CabMoro_S3Rnd1=line[52].split(":")
        CabMoro_Sr1Rnd1=line[53].split(":")
        CabMoro_Sr2Rnd1=line[54].split(":")
        CabMoro_Sr3Rnd1=line[55].split(":")
        Chica1_1=line[56].split(":")
        Chica1_10=line[57].split(":")
        Chica1_11=line[58].split(":")
        Chica1_12=line[59].split(":")
        Chica1_13=line[60].split(":")
        Chica1_14=line[61].split(":")
        Chica1_2=line[62].split(":")
        Chica1_3=line[63].split(":")
        Chica1_4=line[64].split(":")
        Chica1_5=line[65].split(":")
        Chica1_6=line[66].split(":")
        Chica1_7=line[67].split(":")
        Chica1_8=line[68].split(":")
        Chica1_9=line[69].split(":")
        Chica5_1=line[70].split(":")
        Chica5_2=line[71].split(":")
        Chica5_3=line[72].split(":")
        Chica5_4=line[73].split(":")
        Chica5_5=line[74].split(":")
        Choy_1=line[75].split(":")
        Choy_10=line[76].split(":")
        Choy_11=line[77].split(":")
        Choy_12=line[78].split(":")
        Choy_13=line[79].split(":")
        Choy_14=line[80].split(":")
        Choy_5=line[81].split(":")
        Choy_6=line[82].split(":")
        Choy_9=line[83].split(":")
        Escon_1=line[84].split(":")
        Escon_2=line[85].split(":")
        Escon_3=line[86].split(":")
        Escon_4=line[87].split(":")
        Escon_5=line[88].split(":")
        Escon_6=line[89].split(":")
        Escon_7=line[90].split(":")
        Escon_8=line[91].split(":")
        Escondido2=line[92].split(":")
        Gallinas_B4POG=line[93].split(":")
        Gallinas_G3POG=line[94].split(":")
        Gallinas_H3POG=line[95].split(":")
        Japlin1=line[96].split(":")
        Japones3B=line[97].split(":")
        Japones_T5769_S24=line[98].split(":")
        Japones_T5770_S25=line[99].split(":")
        Japones_japo1_S22=line[100].split(":")
        Japones_japo2_S23=line[101].split(":")
        Jineo_1=line[102].split(":")
        Jineo_2=line[103].split(":")
        Jineo_3=line[104].split(":")
        Jineo_4=line[105].split(":")
        Jineo_5=line[106].split(":")
        Jos_1_S14=line[107].split(":")
        Jos_T4550_S16=line[108].split(":")
        Jos_T4555_S18=line[109].split(":")
        Jos_T4587_S17=line[110].split(":")
        Jos_T4588_S15=line[111].split(":")
        Mante_T6151_S25=line[112].split(":")
        Mante_T6152_S24=line[113].split(":")
        Mante_T6153_S28=line[114].split(":")
        Mante_T6154_S29=line[115].split(":")
        Mante_T6155_S26=line[116].split(":")
        Mante_T6156_S27=line[117].split(":")
        Mante_T6157_S22=line[118].split(":")
        Mante_T6158_S20=line[119].split(":")
        Mante_T6159_S23=line[120].split(":")
        Mante_T6160_S21=line[121].split(":")
        Micos_2Boro=line[122].split(":")
        Molino1B=line[123].split(":")
        Molino2B=line[124].split(":")
        Molino4B=line[125].split(":")
        Molino6=line[126].split(":")
        Molino_10b=line[127].split(":")
        Molino_11a=line[128].split(":")
        Molino_12a=line[129].split(":")
        Molino_13b=line[130].split(":")
        Molino_14a=line[131].split(":")
        Molino_15b=line[132].split(":")
        Molino_2a=line[133].split(":")
        Molino_7a=line[134].split(":")
        Molino_9b=line[135].split(":")
        Molinos_2Boro=line[136].split(":")
        Molinos_D3POG=line[137].split(":")
        Montecillos_T5775_S24=line[138].split(":")
        Montecillos_T5776_S27=line[139].split(":")
        Montecillos_T5777_S19=line[140].split(":")
        Montecillos_T5779_S12=line[141].split(":")
        Montecillos_T5780_S15=line[142].split(":")
        Montecillos_T5781_S18=line[143].split(":")
        Montecillos_T5782_S21=line[144].split(":")
        Montecillos_T5785_S22=line[145].split(":")
        Montecillos_T5786_S29=line[146].split(":")
        Pach_11=line[147].split(":")
        Pach_12=line[148].split(":")
        Pach_14=line[149].split(":")
        Pach_15=line[150].split(":")
        Pach_17=line[151].split(":")
        Pach_3=line[152].split(":")
        Pach_7=line[153].split(":")
        Pach_8=line[154].split(":")
        Pach_9=line[155].split(":")
        Pachon2B=line[156].split(":")
        Pachon_1Gross=line[157].split(":")
        Pachon_2Gross=line[158].split(":")
        Pachon_3Gross=line[159].split(":")
        Pachon_6Boro=line[160].split(":")
        Pachon_E2POG=line[161].split(":")
        Pachon_F2POG=line[162].split(":")
        Pachon_G2POG=line[163].split(":")
        Pachon_H2POG=line[164].split(":")
        Pachon_ref=line[165].split(":")
        Palmaseca_T5512_S13=line[166].split(":")
        Palmaseca_T5514_S8=line[167].split(":")
        Palmaseca_T5515_S17=line[168].split(":")
        Palmaseca_T5517_S9=line[169].split(":")
        Palmaseca_T5518_S10=line[170].split(":")
        Palmaseca_T5519_S11=line[171].split(":")
        Palmaseca_T5522_S30=line[172].split(":")
        Palmaseca_T5523_S11=line[173].split(":")
        Peroles_10_T5644_S18=line[174].split(":")
        Peroles_11_T5648_S19=line[175].split(":")
        Peroles_1_S10=line[176].split(":")
        Peroles_2_S11=line[177].split(":")
        Peroles_3_S12=line[178].split(":")
        Peroles_6_T5643_S14=line[179].split(":")
        Peroles_7_T5645_S15=line[180].split(":")
        Peroles_8_T5649_S16=line[181].split(":")
        Peroles_9_T5647_S17=line[182].split(":")
        Rascon1=line[183].split(":")
        Rascon2=line[184].split(":")
        Rascon3=line[185].split(":")
        Rascon3B=line[186].split(":")
        Rascon4=line[187].split(":")
        Rascon5=line[188].split(":")
        Rascon_12=line[189].split(":")
        Rascon_13=line[190].split(":")
        Rascon_15=line[191].split(":")
        Rascon_16=line[192].split(":")
        Rascon_2=line[193].split(":")
        Rascon_4=line[194].split(":")
        Rascon_6=line[195].split(":")
        Rascon_8=line[196].split(":")
        Sabinos_T3076_S26=line[197].split(":")
        Sabinos_T3093_S27=line[198].split(":")
        Subter_1POG=line[199].split(":")
        Subter_2POG=line[200].split(":")
        Subter_3POG=line[201].split(":")
        Subter_4POG=line[202].split(":")
        Subter_5POG=line[203].split(":")
        Subter_6POG=line[204].split(":")
        Subter_7POG=line[205].split(":")
        Subter_8POG=line[206].split(":")
        T5879_Tigre_S31=line[207].split(":")
        T5882_Tigre_S30=line[208].split(":")
        Tigre2=line[209].split(":")
        Tigre_T5881_S20=line[210].split(":")
        Tigre_T5883_S23=line[211].split(":")
        Tigre_T5884_S26=line[212].split(":")
        Tigre_T5885_S28=line[213].split(":")
        Tigre_T5890_S14=line[214].split(":")
        Tigre_T5891_S1=line[215].split(":")
        Tigre_T5893_S2=line[216].split(":")
        Tinaja1B=line[217].split(":")
        Tinaja2B=line[218].split(":")
        Tinaja3=line[219].split(":")
        Tinaja3B=line[220].split(":")
        Tinaja4=line[221].split(":")
        Tinaja4B=line[222].split(":")
        Tinaja_11=line[223].split(":")
        Tinaja_12=line[224].split(":")
        Tinaja_1Boro=line[225].split(":")
        Tinaja_1Gross=line[226].split(":")
        Tinaja_2=line[227].split(":")
        Tinaja_2Gross=line[228].split(":")
        Tinaja_3=line[229].split(":")
        Tinaja_3Gross=line[230].split(":")
        Tinaja_5=line[231].split(":")
        Tinaja_6=line[232].split(":")
        Tinaja_B=line[233].split(":")
        Tinaja_C=line[234].split(":")
        Tinaja_D=line[235].split(":")
        Tinaja_E=line[236].split(":")
        Toro_1_S19=line[237].split(":")
        Toro_2_S20=line[238].split(":")
        Toro_3_S21=line[239].split(":")
        Vanquez3=line[240].split(":")
        Vasquez_V10_S9=line[241].split(":")
        Vasquez_V3_S3=line[242].split(":")
        Vasquez_V5_S4=line[243].split(":")
        Vasquez_V6_S5=line[244].split(":")
        Vasquez_V7_S6=line[245].split(":")
        Vasquez_V8_S7=line[246].split(":")
        Vasquez_V9_S8=line[247].split(":")
        Yerbaniz_T5800_S7=line[248].split(":")
        Yerbaniz_T5802_S16=line[249].split(":")
        Yerbaniz_T5804_S3=line[250].split(":")
        Yerbaniz_T5805_S25=line[251].split(":")
        Yerbaniz_T5806_S4=line[252].split(":")
        Yerbaniz_T5808_S5=line[253].split(":")
        Yerbaniz_T5809_S6=line[254].split(":")
        bb=''
        bb=len(max(line3, key=len)) #take the longest indel
        line4=line[4].split(",")
        cc=''
        cc=len(max(line4, key=len)) # take the longest indel
        
        if int(bb)>1: #Check reference base to ensure that it is not longer than one, if it is that means an indel was found
            if line[0] not in hashy3:
                hashy3[line[0]]=[]
                #print bb
                for rr in range (int(line[1])-10,int(line[1])+int(bb)+10): # If a line has a deletion relative to the reference we add the length of the deletion, this also takes 10 bp +/- these cooridinates
                    hashy3[line[0]].append(rr)
            else:
                for rr in range (int(line[1])-10,int(line[1])+int(bb)+10):
                    hashy3[line[0]].append(rr)

        if int(cc)>1:#Check alt base to ensure that it is not longer than one, if it is that means an indel was found
                if line[0] not in hashy3:
                    hashy3[line[0]]=[]
                    for rr in range (int(line[1])-10,int(line[1])+int(cc)+10): # If a line has a deletion relative to the reference we add the length of the deletion, this also takes 10 bp +/- these cooridinates
                        hashy3[line[0]].append(rr)
                else:
                    for rr in range (int(line[1])-10,int(line[1])+int(cc)+10): # If a line has a deletion relative to the reference we add the length of the deletion, this also takes 10 bp +/- these cooridinates
                        hashy3[line[0]].append(rr)

keys3=hashy3.keys() #gets scaffolds

#for i in keys3:
#    for j in range(0,len(hashy3[i])):
#        a=''
#        b=''
#        a=int(hashy3[i][j])-10
#        b=int(hashy3[i][j])+11
#        for k in (range(a,b)):
#            hashy3[i].append(k)

#hashy4={}
#counter=0
#for line in open(sys.argv[2]):    # This opens the masked sites file, because
#    line = line.strip().split('\t')
#    if counter>0:
#        if line[0] not in hashy4:
#            hashy4[line[0]]=[]
#            for u in range(int(line[1]),int(line[2])+1):
#                hashy4[line[0]].append(u)
#        else:
#            for u in range(int(line[1]),int(line[2])+1):
#                hashy4[line[0]].append(u)
#    counter+=1
    
#keys4=hashy4.keys()

#all_keys=set(keys4+keys3)

print str('sequence')+'\t'+str('begin')+'\t'+str('end')            

for i in keys3: #all_keys
    all_coordinates=[]
    if i in hashy3:
        all_coordinates=hashy3[i]
    myset=()
    mynewlist=[]
    myset = set(all_coordinates)
    mynewlist = list(myset)
    mynewlist.sort(key=int)
    ranges = []
    for k, g in groupby(enumerate(mynewlist), lambda (i,x):i-x): #this takes the indels+the masked sites and makes intervals for exclusion that are suited for vcftools
        group = map(itemgetter(1), g)
        #ranges.append((group[0], group[-1]))
        print str(i)+'\t'+str(int(group[0])-1)+'\t'+str(group[-1])
