import os
import sys

obs = int(sys.argv[2])
date = str(sys.argv[1])

os.system('mkdir /astro/mwaeor/MWA/data/%s/%s' %(str(obs), date))

f1 = 'rts_patch.sh'
fname2 = open('rts_patch_%s.sh' %(str(obs)), 'w')
chunks = open(f1).read().split('\n')
for c in chunks:
        test_sub = '1194628776'
        res = [i for i in range(len(c)) if c.startswith(test_sub, i)]
        if len(res) > 0:
                for r in range(len(res)):
                        if r > 0:
                                indx2 = res[r-1]+11
                                indx3 = res[r]
                                indx4 = res[r] + 10
                                line = line[:indx2]+str(obs)+c[indx4:]

                        else:
                                indx0 = res[r]
                                indx1 = res[r] + 10
                                line = c[:indx0]+str(obs)+c[indx1:]
                fname2.write(line+'\n')
        else:
                fname2.write(c+'\n')
fname2.close()

cmd = 'mv rts_patch_%s.sh /astro/mwaeor/MWA/data/%s/%s' %(str(obs), str(obs), date)
os.system(cmd)

#Uncomment if you need to add flag additional tiles; need to create flagged_tiles.txt
#cmd2 = 'cp flagged_tiles.txt /astro/mwaeor/MWA/data/%s/%s' %(str(obs), date)
#os.system(cmd2)
