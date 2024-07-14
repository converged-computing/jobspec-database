#!/usr/bin/perl

# This is only meant to be used with a split sample sheet file

if (scalar(@ARGV) < 4) {
    print STDERR "Usage: $0 <runinfo file> <sample sheet> <run folder> <output folder>\n";
    exit(1);
}

$runinfo_file = $ARGV[0];
$samplesheet = $ARGV[1];
$runfolder = $ARGV[2];
$outfolder = $ARGV[3];

$mismatch = 1;

open($ri,"<$runinfo_file");
while (<$ri>) {
    if ($_ =~ /\<Read Number=\"(\d)\" NumCycles=\"(\d+)\"/) {
        $runinfo{$1} = $2;
    }
}
close($ri);

open($ss,"<$samplesheet") or die "Cannot open sample sheet: $samplesheet\n";
<$ss>;
<$ss>;
$base_mask = "";
$firstline=0;
while (<$ss>) {
    chomp;
    @data = split(/,/,$_,20);

    $lane = $data[0];
    $index1 = $data[5];
    $index2 = $data[7];

    if ($index1 ne "" && length($index1) > $runinfo{2} && exists $runinfo{2}) {
        print STDERR "Error: Index $index1 is too long. Should be $runinfo{2}bp or less.\n";
        exit(1);
    }

    if ($index2 ne "" && length($index2) > $runinfo{3} && exists $runinfo{3}) {
        print STDERR "Error: Index $index2 is too long. Should be $runinfo{3}bp or less.\n";
        exit(1);
    }

    if ($firstline == 0) {
        $project = $data[8];
        $runtype = scalar(@data) < 11 ? "illumina" : lc($data[10]);
        $projectfolder = "$outfolder/$project";

        $mismatch = scalar(@data) < 12 || $data[11] eq "" ? 1 : $data[11];

        if ($runtype eq "illumina") {
            if (exists $runinfo{1}) {
                $base_mask .= "y".($runinfo{1}-1)."n";
            }

            if (exists $runinfo{2}) {
                if ($index1 ne "") {
                    $n_num = $runinfo{2}-length($index1);
                    if ($n_num > 0) {
                        $base_mask .= ",i".length($index1)."n$n_num";
                    } else {
                        $base_mask .= ",i".length($index1);
                    }
                } else {
                    $base_mask .= ",n$runinfo{2}";
                }
            }

            if (exists $runinfo{3}) {
                if ($index2 ne "") {
                    $n_num = $runinfo{3}-length($index2);
                    if ($n_num > 0) {
                        $base_mask .= ",i".length($index2)."n$n_num";
                    } else {
                        $base_mask .= ",i".length($index2);
                    }
                } else {
                    $base_mask .= ",n$runinfo{3}";
                }
            }

            if (exists $runinfo{4}) {
                $base_mask .= ",y".($runinfo{4}-1)."n";
            }

        } elsif ($runtype eq "10x") {
              $base_mask = "y26,i8,y98";
        }
          
        $firstline = 1;
    }


    if ($indices{$lane}{$index1}{$index2} == 1) {
        print STDERR "Error: $index1".($index2 ne "" ? ",$index2" : "")." is seen more than once in lane $lane.\n";
        exit(1);
    } elsif ($index1 ne "") {
        $indices{$lane}{$index1}{$index2} = 1;
    }
}

#print "$base_mask\n";

#print <<EOF;
#bcl2fastq \\
#--sample-sheet $samplesheet \\
#--runfolder-dir $runfolder \\
#--output-dir $outfolder \\
#--stats-dir $outfolder/Stats \\
#--reports-dir $outfolder/Reports \\
#--create-fastq-for-index-reads \\
#--ignore-missing-positions \\
#--ignore-missing-controls \\
#--ignore-missing-filter \\
#--ignore-missing-bcls \\
#--barcode-mismatches $mismatch \\
#--loading-threads 2 \\
#--processing-threads 14 \\
#--writing-threads 2 \\
#--use-bases-mask $base_mask
#EOF

$command = "bcl2fastq --sample-sheet $samplesheet --runfolder-dir $runfolder --output-dir $outfolder --stats-dir $projectfolder/Stats --reports-dir $projectfolder/Reports --ignore-missing-positions --ignore-missing-controls --ignore-missing-filter --ignore-missing-bcls --barcode-mismatches $mismatch --loading-threads 4 --processing-threads 28 --writing-threads 4 --use-bases-mask $base_mask";

print "$command\n";

$ret = system($command);
exit($ret/256);

#system("sbatch run_bcl2fastq.slurm $samplesheet $runfolder $outfolder $projectfolder $mismatch $base_mask");
#print("sbatch run_bcl2fastq.slurm $samplesheet $runfolder $outfolder $projectfolder $mismatch $base_mask\n");
