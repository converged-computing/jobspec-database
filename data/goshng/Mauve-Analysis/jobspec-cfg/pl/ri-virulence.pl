#!/opt/local/bin/perl -w
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
use strict;
use warnings;
use Getopt::Long;
use Pod::Usage;

$| = 1; # Do not buffer output

my $VERSION = 'ri-virulence.pl 1.0';

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
            'xml=s',
            'xmfa=s',
            'pairm=s',
            'pairs=s',
            'ri=s',
            'ingene=s',
            'samplesize=i',
            'out=s',
            '<>' => \&process
            ) or pod2usage(2);
pod2usage(1) if $help;
pod2usage(-exitstatus => 0, -verbose => 2) if $man;


require "pl/sub-ingene.pl";
require "pl/sub-error.pl";
require "pl/sub-array.pl";
require "pl/sub-newick-parser.pl";
require "pl/sub-xmfa.pl";
require "pl/sub-ri.pl";
require "pl/sub-simple-parser.pl";

################################################################################
## COMMANDLINE OPTION PROCESSING
################################################################################

my $outfile;

if (exists $params{out})
{
  open ($outfile, ">", $params{out}) or die "cannot open > $params{out}: $!";
}
else
{
  $outfile = *STDOUT;   
}

if ($cmd eq "rimap")
{
  unless (exists $params{ri}
          and exists $params{ingene}
          and exists $params{xml}
          and exists $params{xmfa})
  {
    &printError("$cmd requires options -ri, -ingene, and -xml");
  }
}
elsif ($cmd eq "heatmap")
{
  unless (exists $params{ri}
          and exists $params{ingene}
          and exists $params{xml})
  {
    &printError("$cmd requires options -ri, -ingene, and -xml");
  }
}
elsif ($cmd eq "list")
{
  unless (exists $params{ri}
          and exists $params{ingene}
          and exists $params{xml})
  {
    &printError("$cmd requires options -ri, -ingene, and -xml");
  }
}

################################################################################
## DATA PROCESSING
################################################################################

my $tag;
my $content;
my %recedge;
my $itercount = 0;
my $blockLength;
my $blockidForProgress;
my $speciesTree = get_species_tree ("$params{xml}.1");
my $clonaloriginsamplesize = get_sample_size ("$params{xml}.1");
my $numberTaxa = get_number_leave ($speciesTree);
my $tree = maNewickParseTree ($speciesTree);
# maNewickPrintTree ($tree);
my $numberLineage = 2 * $numberTaxa - 1;
my $sizeOfArray = $numberLineage * $numberLineage;

################################################################################
# Select pairs of recombinant edges
################################################################################

# A few matrices are prepared. pairM contains all possible pairs of recombinant
# edges. pairM2 has pairs that do change tree topology. pairM3 has pairs that do
# not change tree topology. pairM4 has only particular pairs.
# my $pairM = maNewicktFindRedEdge ($tree);
# printSquareMatrix ($pairM, $numberLineage);
# print "-------\n-------\n";
# my $pairM2 = maNewicktFindRedEdgeChangeTopology ($tree);
# printSquareMatrix ($pairM2, $numberLineage);
# print "-------\n-------\n";
# my $pairM3 = maNewicktFindRedEdgeNotChangeTopology ($tree);
# printSquareMatrix ($pairM3, $numberLineage);
# print "-------\n-------\n";
# my $pairM4 = maNewicktFindRedEdgePair ($tree, "0,3:0,4:1,3:1,4");
# printSquareMatrix ($pairM4, $numberLineage);
# exit;
# Now, I have pairM. I need to use only those pairs to compute recombination
# intensity. 

################################################################################
# Find coordinates of the reference genome.
################################################################################

sub test ($$$);
sub getRI1Genes ($$$);
sub getRI1Gene ($$$$$);
sub getPairs ($);

# rimap is the new approach.
if ($cmd eq 'rimap')
{
  my @genes = maIngeneParseBlock ($params{ingene}); 
  maIngenePrintBlock ("a", \@genes);

  my $pairM; 
  if ($params{pairm} eq 'all')
  {
    $pairM = maNewicktFindRedEdge ($tree);
  }
  elsif ($params{pairm} eq 'topology')
  {
    $pairM = maNewicktFindRedEdgeChangeTopology ($tree);
  }
  elsif ($params{pairm} eq 'notopology')
  {
    $pairM = maNewicktFindRedEdgeNotChangeTopology ($tree);
  }
  elsif ($params{pairm} eq 'pair')
  {
    $pairM = maNewicktFindRedEdgePair ($tree, $params{pairs});
  }
  else
  {
    die "-pairm options must be one of all, topology, notopology, and pair";
  }
  my @blockStart = maXmfaGetBlockStart ($params{xmfa});
  maRiGetGenesUsingRimapDirectory (\@genes, $params{ri}, \@blockStart, $pairM, $numberLineage, $clonaloriginsamplesize);
  maIngenePrintBlockRi ($outfile, \@genes);
}
elsif ($cmd eq 'list')
{
  # I need to list virulence genes or genes with some noticeable recombinant
  # edges. What should I look at for the noticeable recombinant edges? 
  # We find fraction of a gene where posterior
  # probability of recombinant edges is above a threshold value.
  # I want to ask if I can find virulence genes more associated with a
  # particular recombinant edge type. 
  # We also change the threshold values from 0.01 to 0.99.
  my @genes = maIngeneParseBlock ($params{ingene}); 
  for (my $thresholdID = 50; $thresholdID < 100; $thresholdID += 10)
  {
    my $threshold = $thresholdID / 100 * $clonaloriginsamplesize;
    my $numGene = scalar(@genes);

    ####################################################
    # To deal with genes in multiple blocks
    # The code is similar to that of sub-ri.pl.
    my $indexOfSegment = 1;
    my $baseGenename;
    my $existNextSegment = 0;
    my $lengthAllSegment = 0;
    # my $gBase;
    my @A; # for average recombination probability
    my @B;
    my @C;
    for (my $i = 0; $i < scalar @genes; $i++)
    {
      my $g = $genes[$i];

      $existNextSegment = 0;
      if ($g->{gene} =~ /(.+)\.(\d+)$/)
      {
        $baseGenename = $1;
        $indexOfSegment = $2;
        if ($i+1 <= $#genes)
        {
          my $g2 = $genes[$i+1];
          my $nextIndexOfSegment = $indexOfSegment + 1;
          if ($g2->{gene} eq "$baseGenename.$nextIndexOfSegment")
          {
            $existNextSegment = 1;
          }
        }
      }
      else
      {
        $baseGenename = $g->{gene};
        $indexOfSegment = 1;
      }
      if ($indexOfSegment == 1)
      {
        # $gBase = $g;
        # $gBase->{gene} = $baseGenename;
        @A = (0) x $sizeOfArray;
        @B = (0) x $sizeOfArray;
        @C = (0) x $sizeOfArray;
        # $ri1PerGene = 0;
        $lengthAllSegment = $g->{blockEnd} - $g->{blockStart} + 1;
      }
      else
      {
        $g->{segment} = 1; # Do not print segmental gene.
        $lengthAllSegment += ($g->{blockEnd} - $g->{blockStart} + 1);
      }
      ####################################################

      my $rifile = "$params{ri}/$g->{blockidGene}";
      my $l;
      open RIMAP, $rifile or die "cannot open < $rifile $!";
      for (my $i = 0; $i < $g->{blockStart}; $i++)
      {
        $l = <RIMAP>;
      }
      for (my $i = $g->{blockStart}; $i <= $g->{blockEnd}; $i++)
      {
        $l = <RIMAP>;
        chomp $l;
        my @e = split /\t/, $l;
        shift @e;
        unless (scalar (@e) == $sizeOfArray)
        {
          die "The $i-th line of $rifile does not have $sizeOfArray elements";
        }
        # Increase elements whose values are above the threshold.
        @A = map {$A[$_] + $e[$_]/$clonaloriginsamplesize} 0..$#e;
        @B = map { if ($e[$_] > $threshold) {$B[$_] + 1} else {$B[$_]}} 0..$#e;
        @C = map { if ($e[$_] > $threshold) {$C[$_] + $e[$_]/$clonaloriginsamplesize} else {$C[$_]}} 0..$#e;
      }

      if ($existNextSegment == 0)
      {
        # $gBase->{ri} = $ri1PerGene / ($lengthAllSegment * $sampleSize);

        # Compute the fraction by normalizing values by the length of the genes.
        my $totalLength = $g->{end} - $g->{start} + 1;
        my $lengthAllSegmentPercent = $lengthAllSegment/$totalLength;
        my @B1 = map { $B[$_] / $totalLength } 0..$#B;
        my @B2 = map { $B[$_] / $lengthAllSegment } 0..$#B;
        my @C1 = map { if ($B[$_] > 0) {$C[$_] / ($B[$_])} else {0} } 0..$#C;
        my @A1 = map { $A[$_] / $lengthAllSegment } 0..$#A;

        # print $outfile "$gBase->{gene}";
        print $outfile "$baseGenename";
        print $outfile "\t$totalLength";
        print $outfile "\t$lengthAllSegment";
        print $outfile "\t$lengthAllSegmentPercent";
        print $outfile "\t$thresholdID\t";
        # Total length: $totalLength = $g->{end} - $g->{start};
        # Length in the core genome: $lengthAllSegment
        # %-in-core-genome: $lengthAllSegment/$totalLength 
        # Threshold: $thresholdID
        # Length-segment-above-the-threshold: @B
        # %-the-segment-wrt-total-length: @B1
        # %-the-segment-wrt-length-in-core-genome: @B2
        # Average posterior probability of the segment with prob. above the threshold: @C1
        # Average posterior probability of the segment: @A1
        print $outfile (join ("\t", @B));
        print $outfile "\t";
        print $outfile (join ("\t", @B1));
        print $outfile "\t";
        print $outfile (join ("\t", @B2));
        print $outfile "\t";
        print $outfile (join ("\t", @C1));
        print $outfile "\t";
        print $outfile (join ("\t", @A1));
        print $outfile "\n";
        print STDERR "Reading gene $i - $thresholdID\r";
      }
    }
    print STDERR "                                                 \r";
  }

  # my @genes = parse_in_gene ($params{ingene}); 
  # my @pairSourceDestination = getPairs ($params{pairs});
  # getRI1Genes (\@genes, $params{ri}, \@pairSourceDestination);
}
elsif ($cmd eq 'heatmap')
{
  # A rimap contains an array (a matrix) for each position along blocks.
  # We sum the rimap for the region spanning a gene.
  # Divide each element of the array by the length of the gene and the sample
  # size. Let's call this array A.
  # We repeat the procedure for another gene to create array B. 
  # Sum the two arrays A and B. Note do not divide this by 2 because we still
  # have more arrays from the rest of genes.
  # Repeat the procefure for all of the rest of genes to create arrays.
  # Sum all of the arrays and divide all of the elements of the result array by
  # the number of genes.
  my @A = (0) x $sizeOfArray;
  my @genes = maIngeneParseBlock ($params{ingene}); 
  # maIngenePrintBlock ("a", \@genes);
  my $numGene = scalar(@genes);
  for (my $i = 0; $i < scalar @genes; $i++)
  {
    my $g = $genes[$i];
    my @B = (0) x $sizeOfArray;
    my $rifile = "$params{ri}/$g->{blockidGene}";
    my $l;
    open RIMAP, $rifile or die "cannot open < $rifile $!";
    for (my $i = 0; $i < $g->{blockStart}; $i++)
    {
      $l = <RIMAP>;
    }
    for (my $i = $g->{blockStart}; $i <= $g->{blockEnd}; $i++)
    {
      $l = <RIMAP>;
      chomp $l;
      my @e = split /\t/, $l;
      shift @e;
      unless (scalar (@e) == $sizeOfArray)
      {
        die "The $i-th line of $rifile does not have $sizeOfArray elements";
      }
      @B = map { $B[$_] + $e[$_] } 0..$#B;
    }
    @B = map { $B[$_] / ($g->{blockEnd} - $g->{blockStart} + 1) } 0..$#B;
    @B = map { $B[$_] / $clonaloriginsamplesize } 0..$#B;
    @A = map { $A[$_] + $B[$_] } 0..$#A;
    close RIMAP;
    print STDERR "Reading gene $i/$numGene\r";
  }
  print STDERR "                                                  \r";
  @A = map { $A[$_] / scalar (@genes) } 0..$#A;
  print $outfile (join ("\t", @A));
  print $outfile "\n";
}

close $outfile;
exit;
################################################################################
## END OF DATA PROCESSING
################################################################################

################################################################################
## FUNCTION DEFINITION
################################################################################



sub getPairs ($)
{
  my ($s) = @_;
  my @v;

  my @e = split /:/, $s;
  for my $element (@e)
  {
    push @v, [ split /,/, $element ]; 
  }
  return @v;
}

sub getRI1Genes ($$$)
{
  my ($genes, $ri1map, $pairSourceDestination) = @_;
  open OUT, ">$params{out}" or die "Could not open $params{out} $!";
  my $geneSize = scalar (@{$genes});
  for (my $i = 0; $i < $geneSize; $i++)
  {
    my $h = $genes->[$i];
    my @v = getRI1Gene ($ri1map, $h->{gene}, $h->{start}, $h->{end}, $pairSourceDestination);
    print OUT "$h->{gene}\t";
    print OUT "$h->{start}\t";
    print OUT "$h->{end}\t";
    print OUT "$h->{strand}\t";
    print OUT "$h->{block}\t";
    print OUT "$h->{blockstart}\t";
    print OUT "$h->{blockend}\t";
    print OUT "$h->{genelength}\t";
    print OUT "$h->{proportiongap}\t";
    for (my $j = 0; $j <= $#v; $j++)
    {
      print OUT "\t$v[$j]";
    }
    print OUT "\n";
    print STDERR "Genes $i/$geneSize done ...\r";
  }

  close OUT;
}

# 
sub getRI1Gene ($$$$$)
{
  my ($ri1map, $genes, $start, $end, $pairSourceDestination) = @_;
  my $line;
  

  open RI1MAP, $ri1map or die "Could not open $ri1map $!";
  for (my $i = 1; $i < $start; $i++) 
  {
    $line = <RI1MAP>;
  }

  my @valuePerGene = (0) x $#$pairSourceDestination;
  my $ri1PerGene = 0;
  for my $siteI ($start .. $end) 
  {
    $line = <RI1MAP>;
    chomp $line;
    my @e = split /\t/, $line;
    my $position = $e[0];

    die "$position is not between $start and $end : $line"
      unless $start <= $position and $position <= $end;
   
    my $valuePerSite = 0; 
    for (my $sourceI = 0; $sourceI < $numberLineage; $sourceI++)
    {
      for (my $destinationJ = 0; $destinationJ < $numberLineage; $destinationJ++)
      {
        my $v = $e[1 + $sourceI * $numberLineage + $destinationJ];
        die "Negative means no alignemnt in $position" if $v < 0;
        $valuePerSite += $v;
      }
    }
    for my $i ( 0 .. $#$pairSourceDestination) {
      my $sourceI = $pairSourceDestination->[$i][0];
      my $destinationJ = $pairSourceDestination->[$i][1];
      my $v = $e[1 + $sourceI * $numberLineage + $destinationJ];
      $valuePerGene[$i] += $v;
    }

    $ri1PerGene += $valuePerSite; 
  }
  $ri1PerGene /= ($end - $start + 1);
  $ri1PerGene /= $clonaloriginsamplesize;
  for my $i ( 0 .. $#$pairSourceDestination) {
    $valuePerGene[$i] /= ($end - $start + 1);
    $valuePerGene[$i] /= $clonaloriginsamplesize;
  }
  push @valuePerGene, $ri1PerGene;
  close RI1MAP;
  return @valuePerGene; 
}

__END__

=head1 NAME

ri-virulence.pl - Compute recombination intensity on genes

=head1 VERSION

v1.0, Wed Oct 26 09:09:09 EDT 2011

=head1 SYNOPSIS

perl ri-virulence.pl heatmap -ri file.ri -ingene file.ingene

Compute recombination intensity heat maps for the genes.

perl ri-virulence.pl rimap -ri file.ri -ingene file.ingene

Compute recombination intensity for the genes.

perl ri-virulence.pl list -ri file.ri -ingene file.ingene

List genes using some threshold.

perl ri-virulence.pl mean -ri file.ri -ingene file.ingene

List genes with recombination probability.

=head1 DESCRIPTION

This PERL script is not only for virulent genes. This is just for genes. 

We wish to find genes with average recombination probability.

I compute average probability of experiencing recombinant edges per site for
genes. A gene is located with respect to the alignment blocks. We know where
they start and end in the blocks. 

Two files are necessary: rimap, and ingene files.
4. sample size (-samplesize) 

We could figure out how many species in the tree using the number of elements
for a site.

A directory would contain files of recombinant edge counts along sites as many
as blocks. I could access them to create edge counts. There are files in 
output/cornellf/3/run-analysis/rimap-2 directory. We could use these. We could
use output/cornellf/3/run-analysis/rimap-2.txt

A set of genes is prepared in an ingene file.  The ingene file contains
locations of genes. There are two positions: one in a genome, and another in
blocks. We need these block locations. How do we get these ingene?
We would need a list of all of the genes from SPY1 genome.
We should locate the genes in the blocks.
We also need a list of virulence genes in the SPY1 genome.
There are menus *convert-gff-ingene* and *locate-gene-in-block* that might be
useful in this work.

=head1 OPTIONS

=over 8

=item B<heatmap>

A command heatmap allows to create a matrix of average recombination
probability.

=item B<ri>

A command ri allows to compute a single value of recombination intensity for
genes.

=item B<-ri> <file>

A recombination intensity 1 map file, which should be created using
recombination-intensity1-map.

=item B<-ingene> <file>

An ingene file.

=item B<-out> <file>

An output file.

=item B<-samplesize> <number>

The sample size of recombinant trees.

=back

=head1 AUTHOR

Sang Chul Choi, C<< <goshng_at_yahoo_dot_co_dot_kr> >>

=head1 BUGS

If you find a bug please post a message mauve-analysis project at codaset dot
com repository so that I can make ri-virulence.pl better.

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
