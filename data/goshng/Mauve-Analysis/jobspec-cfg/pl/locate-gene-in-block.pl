#!/usr/bin/perl
###############################################################################
# Copyright (C) 2011 Sang Chul Choi
#
# This file is part of Mauve Analysis.
# 
# Mauve Analysis is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Mauve Analysis is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Mauve Analysis.  If not, see <http://www.gnu.org/licenses/>.
###############################################################################

#==============================================================================
#   Author: Sang Chul Choi, BSCB @ Cornell University, NY
#
#   File: locate-gene-in-block.pl
#   Date: Fri May  6 23:16:57 EDT 2011
#   Version: 1.0
#==============================================================================

use strict;
use warnings;

use Getopt::Long;
use Pod::Usage;
require "pl/sub-ingene.pl";
require "pl/sub-xmfa.pl";
require "pl/sub-fasta.pl";

$| = 1; # Do not buffer output

my $VERSION = 'locate-gene-in-block.pl 1.0';

my $cmd = "";
sub process {
  my ($a) = @_;
  $cmd = $a;
}

my $man = 0;
my $help = 0;
my %params = ('help' => \$help, 'h' => \$help, 'man' => \$man);        
GetOptions( \%params,
            'help|h',
            'man',
            'verbose',
            'version' => sub { print $VERSION."\n"; exit; },
            'ingene=s',
            'xmfa=s',
            'fna=s',
            'refgenome=i',
            'printseq',
            'out=s',
            '<>' => \&process
            ) or pod2usage(2);
pod2usage(1) if $help;
pod2usage(-exitstatus => 0, -verbose => 2) if $man;

################################################################################
## COMMANDLINE OPTION PROCESSING
################################################################################

my $verbose = 0;
my $ingene;
my $xmfa;
my $refgenome;
my $printseq = 0;
my $out;
my $outfile;
my $fna;

if (exists $params{fna}) {
  $fna = $params{fna};
} else {
  &printError("you did not specify a fna file name");
}

if (exists $params{ingene}) {
  $ingene = $params{ingene};
} else {
  &printError("you did not specify a ingene file name");
}

if (exists $params{xmfa}) {
  $xmfa = $params{xmfa};
} else {
  &printError("you did not specify an xmfa file name");
}

if (exists $params{refgenome}) {
  $refgenome = $params{refgenome};
} else {
  &printError("you did not specify an refgenome file name");
}

if (exists $params{verbose}) {
  $verbose = 1;
}

if (exists $params{printseq}) {
  $printseq = 1;
}

if (exists $params{out})
{
  $out = $params{out};
  open ($outfile, ">", $out) or die "cannot open > $out: $!";
}
else
{
  &printError("you did not specify the output file name");
  # $outfile = *STDOUT;   
}

################################################################################
## DATA PROCESSING
################################################################################

sub locate_gene($$$$$);
sub find_coordinates_refgenome ($$);
sub get_number_block ($);
sub locate_gene_in_block($$$); 
sub locate_sites_gene_in_block($$$$); 

if ($cmd eq "")
{
  locate_gene_in_block($ingene, $xmfa, $refgenome); 
  close $outfile;
}
elsif ($cmd eq "locate")
{
  locate_sites_gene_in_block($ingene, $xmfa, $refgenome, $fna); 
  close $outfile;

  # Copy the output file.
  open OUT, $params{out} or die "cannnot < $params{out} $!";
  open OUT2, ">$params{out}.temp" or die "cannnot > $params{out}.temp $!";
  while (<OUT>)
  {
    print OUT2;
  }
  close OUT2;

  # Remove genes with short lengths.
  open OUT, ">", $params{out} or die "cannnot < $params{out} $!";
  open OUT2, "$params{out}.temp" or die "cannnot < $params{out}.temp $!";
  my $l = <OUT2>;
  my $l2;
  print OUT $l;
  while ($l = <OUT2>)
  {
    my @e = split /\t/, $l;
    my $genename = $e[0];
    my $seqLen = $e[9];
    if ($genename =~ /\.(\d)$/) 
    {
      $l2 = <OUT2>;
      my @e2 = split /\t/, $l2;
      my $genename2 = $e2[0];
      if ($genename2 =~ /\.(\d)$/) 
      {
        $seqLen += $e2[9];
      }
      else
      {
        die "The next gene is not the second.";
      }
    }
    if ($seqLen > 50)
    {
      print OUT $l;
      if ($genename =~ /\.(\d)$/) 
      {
        print OUT $l2;
      }
    }
  }
  close OUT2;
}

exit;

################################################################################
## END OF SCRIPT
################################################################################

=head1 PROCEDURE

  Argument 1: Ingene file
  Argument 2: Core genome in XMFA format
  Argument 3: Reference genome index
  Argument 4: Reference genome in FASTA format

  Read in ingene file. The first four columns are gene name, start in the
  genome, end position in the genome, and strand in the genome. A gene can be in
  multiple blocks. For now, I use only one block. There were 18 cases of genes
  that were in multiple blocks. An output file is also tab-delimited one just
  like the input ingene file. Appended columns are: block ID, start position in
  the block of the gene (0-base), end position (0-base), start position of the
  gene in the reference genome (1-base) considering gene position in the block
  (only part of gene can be in the block), end position of the gene, the
  sequence length of the gene in the block (partial gene length can be shorter
  than its original length), proportion of gaps in the extracted gene from the
  block.

  The following is from the source code to show the order of output file columns:
  $g->{gene} $g->{start} $g->{end} $g->{strand} 
  $g->{blockidGene} $g->{blockStart} $g->{blockEnd}
  $g->{geneStartInBlock} $g->{geneEndInBlock}
  $g->{lenSeq} $g->{gap}

  FIXME: what to do with gene with multiple blocks?

  The gene length in the block is $g->{blockEnd} - $g->{blockStart} + 1.
  Therefore, the gene start at blockStart and end in blockEnd.

=cut
sub locate_sites_gene_in_block($$$$) {
  my ($ingene, $xmfa, $r, $fna) = @_;

  print $outfile "gene\tstart\tend\tstrand";
  print $outfile "\tblockidGene\tblockStart\tblockEnd";
  print $outfile "\tgeneStartInBlock\tgeneEndInBlock"; 
  print $outfile "\tlenSeq\tgap\n";

  my @genes = maIngeneParse ($ingene); 
  my $fnaSequence = maFastaParse ($fna);

  my @blockLocationGenome = maXmfaGetCoordinatesRefgenome ($xmfa, $r);
  if ($verbose == 1)
  {
    for (my $i = 0; $i <= $#blockLocationGenome; $i++)
    {
      my $h = $blockLocationGenome[$i];
      print STDERR "$i\t$h->{start}\t$h->{end}\n";
    }
  }

  for (my $i = 0; $i <= $#genes; $i++)
  {
    # Find the gene type.
    my $gType = 0;
    my $g = $genes[$i];
    die "Gene $g->{gene} has problem in coordinates"
      unless $g->{start} < $g->{end};
    my @blockidGenes;
    for (my $j = 0; $j <= $#blockLocationGenome; $j++)
    {
      $gType = 0;
      my $b = $blockLocationGenome[$j];
      die "Block $b has problem in coordinates"
        unless $b->{start} < $b->{end};
      # Block        |--------------------|
      # Tyep 1:          <---->
      # Type 2: <----------------------------->
      # Type 3:   <------>
      # Type 4:                        <----->
      # Type 5: <-->                           <----->
      if (($b->{start} <= $g->{start} and $g->{start} <= $b->{end}) and
          ($b->{start} <= $g->{end} and $g->{end} <= $b->{end}))
      {
        $gType = 1;
      }
      elsif ($g->{start} < $b->{start} and $b->{end} < $g->{end})
      {
        $gType = 2;
      }
      elsif ($g->{start} < $b->{start} and 
             ($b->{start} <= $g->{end} and $g->{end} <= $b->{end}))
      {
        $gType = 3;
      }
      elsif (($b->{start} <= $g->{start} and $g->{start} <= $b->{end}) and
             $b->{end} < $g->{end})
      {
        $gType = 4;
      }
      elsif ($g->{end} < $b->{start} or $b->{end} < $g->{start})
      {
        $gType = 5;
      }
      else
      {
        die "Impossible type of gene $g->{gene} location w.r.t. block $j (0-base): g_start: $g->{start}, g_end: $g->{end}, b_start: $b->{start}, b_end: $b->{end})";
      }

      if (1 <= $gType and $gType <= 4)
      {
        push @blockidGenes, $j;
      }
    }

    # Find sites of the gene in the block.
    # if ($blockidGene != -1)
    my $geneName = $g->{gene};
    for (my $blockidGeneIndex = 0; $blockidGeneIndex <= $#blockidGenes; $blockidGeneIndex++)
    {
      my $blockidGene = $blockidGenes[$blockidGeneIndex];
      if ($#blockidGenes > 0)
      {
        my $blockidGeneNumber = $blockidGeneIndex + 1;
        $g->{gene} = "$geneName.$blockidGeneNumber"; 
      }

      # 2. Use the DNA sequence of the reference gene to locate sites that
      # are covered by the gene. I could extract DNA sequences to double
      # check the gene is correctly identified.
      # s is the start position at the block.
      # e is the end position.
      # seq is the DNA sequence.
      # gap is the proportion of gaps in the sequence.
      my ($s, $e, $seq, $gap, 
          $geneStartInBlock, 
          $geneEndInBlock,
          $strandBlock) = maXmfaLocateGene ($xmfa, 
                                            $blockidGene + 1,
                                            $r,
                                            $g->{start},
                                            $g->{end},
                                            $g->{strand});

      my $lenSeq = length $seq;
      $g->{blockidGene} = $blockidGene + 1;
      $g->{blockStart} = $s;
      $g->{blockEnd} = $e;
      $g->{lenSeq} = $lenSeq;
      $g->{seq} = $seq;
      $g->{gap} = $gap;
      $g->{geneStartInBlock} = $geneStartInBlock;
      $g->{geneEndInBlock} = $geneEndInBlock;

      unless ($g->{start} == $g->{geneStartInBlock} and $g->{end} == $g->{geneEndInBlock})
      {
        # print $outfile "CHECK\t";
      }
#      if ($lenSeq >= 0)
#      {
        print $outfile "$g->{gene}\t$g->{start}\t$g->{end}\t$g->{strand}";
        print $outfile "\t$g->{blockidGene}\t$g->{blockStart}\t$g->{blockEnd}";
        print $outfile "\t$g->{geneStartInBlock}\t$g->{geneEndInBlock}"; 
        print $outfile "\t$g->{lenSeq}\t$g->{gap}\n";

        if ($verbose == 1) {
          print STDERR "$g->{gene}\t$g->{start}\t$g->{end}\t$g->{strand}\t$g->{blockidGene}\t$s\t$e\n";
          print STDERR "$g->{seq}\n";
        }

        ##########################################################################
        # Make sure the sequence from XMFA and that of Reference genome do
        # match.
        #
        my $geneSequenceFromXMFA = maXmfaDotNumberGetSequence ("$xmfa.$g->{blockidGene}", $r);
        my $partialSequenceFromRefgenome = substr ($fnaSequence,
                                                   $g->{geneStartInBlock}-1,
                                                   $lenSeq);
        my $partialSequenceFromXMFA = substr ($geneSequenceFromXMFA,
                                              $g->{blockStart},
                                              $g->{blockEnd} - $g->{blockStart} + 1);
        $partialSequenceFromXMFA =~ s/-//g;
        if ($strandBlock eq '-')
        {
          $partialSequenceFromXMFA = reverse $partialSequenceFromXMFA;
          $partialSequenceFromXMFA =~ tr/ACGTacgt/TGCAtgca/; 
        }
        unless (lc($partialSequenceFromRefgenome) eq lc($partialSequenceFromXMFA)) 
        {
          die "Sequences from XMFA and Reference genome do not match
               [$partialSequenceFromXMFA]
               [$partialSequenceFromRefgenome]
               gene: $g->{gene}
               start: $g->{start}
               end: $g->{end}
               strand: $g->{strand}
               blockID:$g->{blockidGene}
               blockStart: $g->{blockStart}
               blockEnd: $g->{blockEnd}
               geneStartInBlock: $g->{geneStartInBlock}
               geneEndInBlock: $g->{geneEndInBlock}
               lenSeq: $g->{lenSeq}
               gap: $g->{gap}
               $g->{seq}";
        }
        #
        # Check is done.
        ##########################################################################
#      }
#      else
#      {
#        print STDERR "$g->{gene} was too short ($lenSeq or shorter than 50 base pairs long)\n";
#      }
    }
    unless (@blockidGenes) 
    {
      if ($verbose == 1) 
      {
        print STDERR "$g->{gene} was not found in the blocks\n";
      }
    }
  }
}

sub locate_gene_in_block($$$) {
  my ($ingene, $xmfa, $r) = @_;

  my $ingene2 = "$ingene.temp";
  open OUT, ">$ingene2" or die "cannot open > $ingene2"; 

  my @genes = maIngeneParse ($ingene); 

  my @blockLocationGenome = maXmfaGetCoordinatesRefgenome ($xmfa, $r);
  if ($verbose == 1)
  {
    for (my $i = 0; $i <= $#blockLocationGenome; $i++)
    {
      my $h = $blockLocationGenome[$i];
      print STDERR "$i\t$h->{start}\t$h->{end}\n";
    }
  }
   
  for (my $i = 0; $i <= $#genes; $i++)
  {
    my $h = $genes[$i];
    my $blockidGene = -1;
    for (my $j = 0; $j <= $#blockLocationGenome; $j++)
    {
      my $g = $blockLocationGenome[$j];
      if ($g->{start} <= $h->{start} and $h->{end} <= $g->{end})
      {
        $blockidGene = $j + 1;
        last;
      }
    }

    if ($blockidGene > 0)
    {
      # my $href = $blockLocationGenome[$blockidGene - 1];
      # 2. Use the DNA sequence of the reference gene to locate sites that
      # are covered by the gene. I could extract DNA sequences to double
      # check the gene is correctly identified.
      my ($s, $e, $seq, $gap) = locate_gene ($xmfa, $blockidGene,
                                             $r, 
                                             $h->{start},
                                             $h->{end});
      my $lenSeq = length $seq;
      if ($verbose == 1) {
        print STDERR "$h->{gene}\t$h->{start}\t$h->{end}\t$h->{strand}\t$blockidGene\t$s\t$e\n";
      }
      print OUT "$h->{gene}\t$h->{start}\t$h->{end}\t$h->{strand}\t$blockidGene\t$s\t$e";
      if ($printseq == 1)
      {
        #print OUT "\t$seq\t$lenSeq\t$gap\n";
        print OUT "\t$lenSeq\t$gap\n";
      }
      else
      {
        print OUT "\n";
      }
    }
    else
    {
      if ($verbose == 1) {
        print STDERR "$h->{gene} was not found in the blocks\n";
      }
    }
  }

  close OUT;

  rename $ingene2, $ingene;
}

################################################################################
# Find coordinates of the reference genome.
################################################################################

sub get_number_block ($)
{
  my ($f) = @_;
  my $v = 0;
  open XMFA, $f or die "$f could not be opened";
  while (<XMFA>)
  {
    if (/^=/)
    {
      $v++;
    }
  }
  close XMFA;
  return $v;
}

# Gene position in the block is 0-based location.
sub locate_gene($$$$$)
{
  my ($f, $blockid, $r, $s, $e) = @_;
  my $startGenome;
  my $endGenome;
  my $sequence = "";

  my $v = 1;
  open XMFA, $f or die "Could not open $f";
  while (<XMFA>)
  {
    if (/^=/)
    {
      $v++;
    }

    if (/^>\s+$r:(\d+)-(\d+)/ and $v == $blockid)
    {
      $startGenome = $1;
      $endGenome = $2;
      last;
    }
  }
  die "The gene is not in the block"
    unless $startGenome <= $s and $e <= $endGenome;

  my $line;
  while ($line = <XMFA>)
  {
    chomp $line;
    if ($line =~ /^>/)
    {
      last;
    }
    $sequence .= $line;
  }
  close XMFA;

  my $geneStartBlock = -1;
  my $geneEndBlock = -1;
  my $j = 0;
  my @nucleotides = split //, $sequence;
  for (my $i = 0; $i <= $#nucleotides; $i++)
  {
    if ($nucleotides[$i] eq 'a' 
        or $nucleotides[$i] eq 'c' 
        or $nucleotides[$i] eq 'g' 
        or $nucleotides[$i] eq 't') 
    {
      my $pos = $startGenome + $j;   
      if ($e == $pos && $geneStartBlock > -1)
      {
        $geneEndBlock = $i;
        last;
      }
      if ($s == $pos && $geneStartBlock == -1)
      {
        $geneStartBlock = $i;
      }

      $j++;
    }
  }
  my $lenGene = $geneEndBlock - $geneStartBlock + 1;
  my $geneSequence = substr $sequence, $geneStartBlock, $lenGene;

  my $percentageGap;
  my %count; 
  $count{$_}++ foreach split //, $geneSequence;
  if (exists $count{"-"})
  {
    $percentageGap = $count{"-"} / $lenGene;
  }
  else
  {
    $percentageGap = 0;
  }

  $geneSequence =~ s/-//g;
  return ($geneStartBlock, $geneEndBlock, $geneSequence, $percentageGap); 
}
__END__
=head1 NAME

locate-gene-in-block.pl

=head1 VERSION

locate-gene-in-block.pl 1.0

=head1 SYNOPSIS

perl locate-gene-in-block.pl.pl [-h] [-help] [-man] [-version] [-verbose]
  [locate]
  [-ingene file] 
  [-xmfa core_alignment.xmfa] 
  [-refgenome number] 
  [-fna file] 
  [-out file] 

=head1 DESCRIPTION

locate-gene-in-block.pl locates genes in blocks. 

=head1 OPTIONS

=over 8

=item B<locate>

locate is used when users want to find sites of genes in blocks site-by-site.

=item B<-help> | B<-h>

Print the help message; ignore other arguments.

=item B<-version>

Print program version; ignore other arguments.

=item B<-verbose>

Prints status and info messages during processing.

=item B<-ingene> <file>

An ingene file name.

=item B<-xmfa> <core_alignment.xmfa>

An input core alignment file.

=item B<-fna> <file>

A file in FASTA format with the reference genome sequence.

=item B<-refgenome> <number>

An reference genome.

=item B<-out> <file>

The out file.

=back

=head1 AUTHOR

Sang Chul Choi, C<< <goshng_at_yahoo_dot_co_dot_kr> >>

=head1 BUGS

If you find a bug please post a message rnaseq_analysis project at codaset dot
com repository so that I can make locate-gene-in-block.pl better.

=head1 COPYRIGHT

Copyright (C) 2011 Sang Chul Choi

=head1 LICENSE

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.

=cut


