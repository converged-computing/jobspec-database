#!/projects/tau/packages/python/3.6.0/bin/python3
#Adrian Bubie
#9/9/17

#Part1: Assessing Quality scores

def pull_qual(file, stats_arr):
    
    qs_dist = []
    
    with open(file,'r') as r1:
        NR = 0
        for line in r1:
            NR += 1
            #if NR > 10000:
             #   break

            line = line.strip('\n')
            if NR%4 == 0:
                read_qs = 0
                for j in range(0, len(line)):
                    read_qs += (ord(line[j])-33)
                    stats_arr[j] += (ord(line[j])-33)
                    j += 1
                
                mean_int = int(read_qs/len(stats_arr))
                qs_dist.append(mean_int)
             
            if NR in [1000,5000,100000,1000000,10000000]:
                print("Processing...",str(NR))   
    
    print("Number of Lines: ", NR)
    print("Quality Breakdown for", file)
    headers = ['# Base Pair','Mean Quality Score']
    print('\t'.join(headers))
    
    means = []
    
    if len(stats_arr) > 8:
        for k in range(0,101):
            mean = stats_arr[k]/(NR/4)
            means.append(mean)
            print(str(k)+'\t'+str(mean))
            
    else:
        for k in range(0,8):
            mean = stats_arr[k]/(NR/4)
            means.append(mean)
            print(str(k)+'\t'+str(mean))
     
    return means, qs_dist

            
stats_ar_r1 = [0.0]*101 # Read 1
stats_ar_i1 = [0.0]*8 # Index 1
stats_ar_i2 = [0.0]*8 # Index 2
stats_ar_r2 = [0.0]*101 # Read 2

r1 = '/projects/bgmp/2017_sequencing/1294_S1_L008_R1_001.fastq'
r2 = '/projects/bgmp/2017_sequencing/1294_S1_L008_R4_001.fastq'
i1 = '/projects/bgmp/2017_sequencing/1294_S1_L008_R2_001.fastq'
i2 = '/projects/bgmp/2017_sequencing/1294_S1_L008_R3_001.fastq'
 
mean_r1 = pull_qual(r1,stats_ar_r1)
mean_r2 = pull_qual(r2,stats_ar_r2)
mean_i1 = pull_qual(i1,stats_ar_i1)
mean_i2 = pull_qual(i2,stats_ar_i2)

with open('r1_dist_out.csv', 'a') as out_1:
    for v in mean_r1[1]:
        out_1.write(str(v)+'\n')
    
with open('r2_dist_out.csv', 'a') as out_2:
    for v in mean_r2[1]:
        out_2.write(str(v)+'\n')
    
with open('i1_dist_out.csv', 'a') as out_3:
    for v in mean_i1[1]:
        out_3.write(str(v)+'\n')
    
with open('i2_dist_out.csv', 'a') as out_4:
    for v in mean_i2[1]:
        out_4.write(str(v)+'\n')

with open('r1_bp_out.csv', 'a') as out_1:
    j = 0
    for i in mean_r1[0]:
        out_1.write(str(j)+','+str(i)+'\n')
        j += 1

with open('r2_bp_out.csv', 'a') as out_2:
    j = 0
    for i in mean_r2[0]:
        out_2.write(str(j)+','+str(i)+'\n')
        j += 1

with open('i1_bp_out.csv', 'a') as out_3:
    j = 0
    for i in mean_i1[0]:
        out_3.write(str(j)+','+str(i)+'\n')
        j += 1
        
with open('i2_bp_out.csv', 'a') as out_4:
    j = 0
    for i in mean_i2[0]:
        out_4.write(str(j)+','+str(i)+'\n')
        j += 1
