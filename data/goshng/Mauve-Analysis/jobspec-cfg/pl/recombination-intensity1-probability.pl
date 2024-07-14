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
require "pl/sub-ingene.pl";
require "pl/sub-error.pl";
require "pl/sub-array.pl";
require "pl/sub-simple-parser.pl";
require "pl/sub-newick-parser.pl";
require "pl/sub-xmfa.pl";
require "pl/sub-ri.pl";
require "pl/sub-ps.pl";
require "pl/sub-maf.pl";
require "pl/sub-gbk.pl";

$| = 1; # Do not buffer output
my $VERSION = 'recombination-intensity1-probability.pl 1.0';
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
            'gbk=s',
            'xml=s',
            'xmfa2maf=s',
            'xmfa=s',
            'refgenome=i',
            'block=i',
            'ri1map=s',
            'outdir=s',
            'ri=s',
            'ingene=s',
            'clonaloriginsamplesize=i',
            'out=s',
            '<>' => \&process
            ) or pod2usage(2);
pod2usage(1) if $help;
pod2usage(-exitstatus => 0, -verbose => 2) if $man;

sub getLengthMap ($);
sub drawRI1BlockGenes ($$);
sub getRI1Gene ($$$$$);
sub getPairs ($);

################################################################################
## COMMANDLINE OPTION PROCESSING
################################################################################
my $ri1map;
my $ingene;
my $out; 
my $clonaloriginsamplesize;
my $pairs;
my $verbose = 0;
my $xml;
my $xmfa;
my $refgenome; 
my $block = -1;

if (exists $params{block})
{
  $block = $params{block};
}

if (exists $params{xml})
{
  $xml = $params{xml};
}

if (exists $params{xmfa})
{
  $xmfa = $params{xmfa};
}


if (exists $params{ri1map})
{
  $ri1map = $params{ri1map};
}

if (exists $params{out})
{
  $out = $params{out};
}

if (exists $params{ingene})
{
  $ingene = $params{ingene};
}

if (exists $params{clonaloriginsamplesize})
{
  $clonaloriginsamplesize = $params{clonaloriginsamplesize};
}

if (exists $params{verbose})
{
  $verbose = 1;
}

if (exists $params{refgenome})
{
  $refgenome = $params{refgenome};
}

if ($cmd eq "ps")
{
  unless (exists $params{ingene}
          and exists $params{xmfa}
          and exists $params{clonaloriginsamplesize}
          and exists $params{refgenome}
          and exists $params{xml})
  {
    &printError("Command $cmd requires options -ingene");
  }
}
elsif ($cmd eq "split")
{
  unless (exists $params{xmfa} and exists $params{ri1map}
          and exists $params{outdir})
  {
    &printError("Command $cmd requires options -xmfa, -outdir and -ri1map");
  }
}
elsif ($cmd eq "wiggle")
{
  unless (exists $params{xmfa2maf} and exists $params{gbk}
          and exists $params{xmfa}
          and exists $params{clonaloriginsamplesize}
          and exists $params{refgenome}
          and exists $params{ri}
          and exists $params{xml})
  {
    &printError("Command $cmd requires options -xmfa2maf, and -gbk");
  }
}

################################################################################
## DATA PROCESSING
################################################################################
my $itercount = 0;
my $blockLength;
my $blockidForProgress;
my $speciesTree;
my $numberTaxa;
my $numberLineage; 
my @genes;
if ($cmd eq "ps")
{
  $speciesTree = get_species_tree ("$xml.1");
  $numberTaxa = get_number_leave ($speciesTree);
  $numberLineage = 2 * $numberTaxa - 1;

  ################################################################################
  # Find coordinates of the reference genome.
  ################################################################################

  my $numberBlock = xmfaNumberBlock ($xmfa);
  @genes = maIngeneParseBlock ($ingene); 
  my $lengthTotalBlock = maRiGetLength ($ri1map);
  my @blockStart = maXmfaGetBlockStart ($xmfa);

  if ($block == -1)
  {
    for (my $b = 1; $b <= $numberBlock; $b++)
    {
      maPsDrawRiBlock ($out, $ri1map, \@blockStart, \@genes, $b, $clonaloriginsamplesize);
    }
  }
  else
  {
    maPsDrawRiBlock ($out, $ri1map, \@blockStart, \@genes, $block, $clonaloriginsamplesize);
  }

# maPsDrawRi (\@genes, $ri1map, $lengthTotalBlock);
#drawRI1BlockGenes (\@genes, $ri1map);
}
elsif ($cmd eq "split")
{
  my $n = xmfaNumberBlock ($params{xmfa});
  my @blockSize = peachXmfaBlockSize ($params{xmfa}, $n);
  open RIMAP, $params{ri1map} or die "cannot open < $params{ri1map} $!";
  for (my $i = 0; $i <= $#blockSize; $i++)
  {
    my $l = $blockSize[$i];
    my $blockID = $i + 1;
    open RIMAPOUT, ">$params{outdir}/$blockID" or die "cannot open > $params{outdir}/$blockID $!";
    foreach my $j (1..$l)
    {
      my $line = <RIMAP>;
      print RIMAPOUT $line;
    }
    close RIMAPOUT;
  }
  close RIMAP;
}
elsif ($cmd eq "wiggle")
{
  my $line; 
  $speciesTree = get_species_tree ("$xml.1");
  $numberTaxa = get_number_leave ($speciesTree);
  $numberLineage = 2 * $numberTaxa - 1;
  my $genomeLength = peachGbkLength ($params{gbk});

  my $n = xmfaNumberBlock ($params{xmfa});
  my @block = getBlockConfiguration ($params{refgenome}, $params{xmfa}, $n);
  my @sortedBlock = sort {$$a{start} <=> $$b{start} } @block;

  my $lineZero = "x";
  foreach my $i (1..$numberLineage) 
  {
    foreach my $j (1..$numberLineage) 
    {
      $lineZero .= "\t0";
    }
  }
  $lineZero .= "\n";

  open OUT, ">", $params{out} or die "cannot open > $params{out} $!";
  my $pos = 1;
  foreach my $b (@sortedBlock)
  {
    for (; $pos < $b->{start}; $pos++)
    {
      print OUT "$pos\t$lineZero";
    }

    open RIMAP, "$params{ri}/$b->{block}"
      or die "cannot open < $params{ri}/$b->{block} $!";
    my @nucleotide = split (//, $b->{seq});
    if ($b->{strand} eq '-')
    {
      @nucleotide = reverse (@nucleotide);
    }
    foreach my $c (@nucleotide)
    {
      $line = <RIMAP>;

      unless ($c eq '-')
      {
        print OUT "$pos\t$line";
        $pos++;
      }
    }
    while ($line = <RIMAP>)
    {
      die "The nucleotide's counts and $params{ri}/$b->{block} do not match";
    }
    unless ($pos == $b->{end} + 1)
    {
      die "Current pos and block end do not match: pos ($pos), block end ($b->{end})"; 
    }
    close RIMAP;
  }
  for (; $pos <= $genomeLength; $pos++)
  {
    print OUT "$pos\t$lineZero";
  }

  close OUT;
}

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

sub drawRI1BlockGenes ($$)
{
  my ($genes, $ri1map) = @_;

  my @map;
  my $line;
  my $status = "nomap";
  open MAP, $ri1map or die "could not open $ri1map";
  while ($line = <MAP>)
  {
    chomp $line;
    my @e = split /\t/, $line;
    if ($status eq "nomap")
    {
      if ($e[1] < 0)
      {
        next;
      }
      else
      {
        @map = ();
        $status = "map";
      }
    }
    else
    {
      if ($e[1] < 0)
      {
        $status = "nomap";
        # Generate the figure for the map.
        drawRI1Block (\@map, $ingene, $out);
        #last;
        next;
      }
      else
      {
        push @map, [ @e ];
      }
    }
  }
  close MAP;
}

sub drawRI1Block ($$$)
{
  my ($map, $ingene, $out) = @_;
  my $lengthGenome = 11111111; # FIXME

  ###############
  # Postscript.
  ###############
  my $height = 792;
  my $width = 612;
  my $numberRow = 9;
  my $upperMargin = 60;
  my $lowerMargin = 200;
  my $rightMargin = 20;
  my $leftMargin = 20;
  my $heightRow = (792 - $upperMargin - $lowerMargin) / $numberRow;
  my $heightBar = int($heightRow * 0.5);
  my $tickBar = int($heightRow * 0.05);
  my $xStart = 100;
  my $tickInterval = 100;

  my $startPos = $map->[0][0];
  my $last = $#{$map};
  my $endPos = $map->[$last][0];
  my $binSize = int(($last + 1) / $width + 1);

  # Start to write a postscript file.
  my $outfile = sprintf("$out-%08d.ps", $startPos);
  open OUT, ">$outfile" or die "could not open $outfile $!";

  # Draw the location of the block.
  my $xGenome = $xStart;
  my $yGenome = $height - 30;
  my $xEndGenome = $width - $rightMargin;
  print OUT "newpath\n";
  print OUT "$xGenome $yGenome moveto\n"; 
  print OUT "$xEndGenome $yGenome lineto\n"; 
  print OUT "0 0 0 setrgbcolor\n";
  print OUT "stroke\n"; 
  # Left-end Bar
  my $yUpperBar = $yGenome + 5;
  print OUT "newpath\n";
  print OUT "$xGenome $yGenome moveto\n"; 
  print OUT "$xGenome $yUpperBar lineto\n"; 
  $yUpperBar = $yGenome - 5;
  print OUT "$xGenome $yUpperBar lineto\n"; 
  print OUT "0 0 0 setrgbcolor\n";
  print OUT "stroke\n"; 
  # Right-end Bar
  $yUpperBar = $yGenome + 5;
  print OUT "newpath\n";
  print OUT "$xEndGenome $yGenome moveto\n"; 
  print OUT "$xEndGenome $yUpperBar lineto\n"; 
  $yUpperBar = $yGenome - 5;
  print OUT "$xEndGenome $yUpperBar lineto\n"; 
  print OUT "0 0 0 setrgbcolor\n";
  print OUT "stroke\n"; 
  # Block
  my $lengthGenomeBar = $xEndGenome - $xGenome;
  my $xBlockBar = $startPos/$lengthGenome*$lengthGenomeBar;
  my $xBlock;
  if ($xBlockBar < 1)
  {
    $xBlock = $xGenome + 1;
  }
  else
  {
    $xBlock = int($xGenome + $startPos/$lengthGenome*$lengthGenomeBar);
  }
  my $xEndBlock = int($xGenome + $endPos/$lengthGenome*$lengthGenomeBar);
  $yUpperBar = $yGenome + 5;
  print OUT "newpath\n";
  print OUT "$xBlock $yGenome moveto\n"; 
  print OUT "$xBlock $yUpperBar lineto\n"; 
  $yUpperBar = $yGenome - 5;
  print OUT "$xBlock $yUpperBar lineto\n"; 
  print OUT "1 0 0 setrgbcolor\n";
  print OUT "stroke\n"; 

  # Write the position of the block
  my $locationBlock = "chr1:$startPos-$endPos";
  $yUpperBar = $yGenome + 10;
  print OUT "/Times-Roman findfont 15 scalefont setfont\n";
  print OUT "0 0 0 setrgbcolor\n";
  print OUT "$xGenome $yUpperBar moveto ($locationBlock) show\n";

  # Draw a scale bar
  my $xScalebar = $xStart;
  my $yScalebar = $height - 60;
  print OUT "newpath\n";
  print OUT "$xScalebar $yScalebar moveto\n"; 
  $xScalebar += $tickInterval; 
  print OUT "$xScalebar $yScalebar lineto\n"; 
  print OUT "0 0 0 setrgbcolor\n";
  print OUT "stroke\n"; 
  # Write the scale in base pairs
  my $scaleSize = sprintf ("%d bp", $binSize * $tickInterval);
  print OUT "/Times-Roman findfont 15 scalefont setfont\n";
  print OUT "0 0 0 setrgbcolor\n";
  $yScalebar -= 5;
  $xScalebar += 5;
  print OUT "$xScalebar $yScalebar moveto ($scaleSize) show\n";

  # Colors
  my @setrgbcolors;
  push @setrgbcolors, "0 0 1 setrgbcolor\n";          # Blue
  push @setrgbcolors, "0 1 1 setrgbcolor\n";          # Aqua
  push @setrgbcolors, "0 1 0 setrgbcolor\n";          # Green
  push @setrgbcolors, "1 0 0 setrgbcolor\n";          # Red
  push @setrgbcolors, "1 0.65 0 setrgbcolor\n";       # Orange
  push @setrgbcolors, "0 0 0 setrgbcolor\n";          # Black
  push @setrgbcolors, "0.65 0.16 0.16 setrgbcolor\n"; # Brown
  push @setrgbcolors, "1 1 0 setrgbcolor\n";          # Yellow
  push @setrgbcolors, "0.93 0.51 0.93 setrgbcolor\n"; # Violet

  # Species
  my @speciesNames;
  push @speciesNames, "SDE1";
  push @speciesNames, "SDE2";
  push @speciesNames, "SDD";
  push @speciesNames, "SPY1";
  push @speciesNames, "SPY2";
  push @speciesNames, "SDE";
  push @speciesNames, "SPY";
  push @speciesNames, "SD";
  push @speciesNames, "ROOT";

  # Draw boxes for each species
  my @destinationJOrder = reverse (0,5,1,7,2,8,3,6,4);
  for (my $destinationJIndex = 0; $destinationJIndex < $numberLineage; $destinationJIndex++)
  {
    my $destinationJ = $destinationJOrder[$destinationJIndex];
    my $i = $destinationJIndex;
    my $yStart = $lowerMargin + int ($i * $heightRow);
  
    my $order = $destinationJOrder[$destinationJIndex] + 1;
    print OUT $setrgbcolors[$order-1];

    print OUT "newpath\n";
    print OUT "70 $yStart moveto\n";
    print OUT "10 0 rlineto\n";
    print OUT "0 10 rlineto\n"; 
    print OUT "-10 0 rlineto\n";
    print OUT "closepath fill\n";
    print OUT "stroke\n";
    print OUT "0 0 0 setrgbcolor\n";       # Orange
    print OUT "20 $yStart moveto ($speciesNames[$order-1]) show\n";
  }

  # Draw genes.
  my $yGene = 10;
  my $numberGene = 0;
  for (my $i = 0; $i < scalar @genes; $i++)
  {
    my $rec = $genes[$i];
    if ($startPos < $rec->{start} and $rec->{start} < $endPos)
    {
      $numberGene++;
      if ($numberGene > 5)
      {
        $yGene = 10;
        $numberGene = 0;
      }
      $yGene += 15;
      my $xGene = $xStart + int(($rec->{start} - $startPos) / ($endPos - $startPos) * ($width - $xStart));
      print OUT "newpath\n";
      print OUT "$xGene $yGene moveto\n"; 
      $yGene += 10;
      print OUT "$xGene $yGene lineto\n"; 
      print OUT "1 0 0 setrgbcolor\n";
      print OUT "stroke\n"; 

      print OUT "$xGene $yGene moveto ($rec->{gene}) show\n";
    }
    if ($startPos < $rec->{end} and $rec->{end} < $endPos)
    {
      if ($startPos < $rec->{start} and $rec->{start} < $endPos)
      {
        $yGene -= 10;
      } 
      else
      {
        $numberGene++;
        if ($numberGene > 5)
        {
          $yGene = 10;
          $numberGene = 0;
        }
      }
      my $xGene = $xStart + int(($rec->{end} - $startPos) / ($endPos - $startPos) * ($width - $xStart));
      print OUT "newpath\n";
      print OUT "$xGene $yGene moveto\n"; 
      $yGene += 10;
      print OUT "$xGene $yGene lineto\n"; 
      print OUT "0 0 1 setrgbcolor\n";
      print OUT "stroke\n"; 
      if ($startPos < $rec->{start} and $rec->{start} < $endPos)
      {
        # No code.
      }
      else
      {
        print OUT "$xGene $yGene moveto ($rec->{gene}) show\n";
      }
    }
    if ($rec->{start} < $startPos and $endPos < $rec->{start})
    {
      my $xGene = $xStart; 
      print OUT "$xGene 10 moveto ($rec->{gene}) show\n";
    }

  }

  # Draw recombination probability.
  @destinationJOrder = reverse (0,5,1,7,2,8,3,6,4);
  for (my $destinationJIndex = 0; $destinationJIndex < $numberLineage; $destinationJIndex++)
  {
    my $destinationJ = $destinationJOrder[$destinationJIndex];
    # my $destinationJ = $destinationJIndex;
    # Postscript.
    my $i = $destinationJIndex;
    my $yStart = $lowerMargin + int ($i * $heightRow);

    my @prob = (0) x $numberLineage;  
    my $binIndex = 0;
    my $j = $xStart;
    for (my $p = 0; $p <= $last; $p++)
    {
      unless ($binIndex < $binSize)
      {
        # Postscript.
        my @sortedOrder = sort { $prob[$b] cmp $prob[$a] } 0 .. $#prob;
        for my $k (1..$numberLineage) 
        {
          my $order = $sortedOrder[$k-1] + 1;
          my $barY = $yStart + int($prob[$order-1] * $heightBar);
          print OUT "newpath\n";
          print OUT "$j $yStart moveto\n";
          print OUT "$j $barY lineto\n";

          print OUT $setrgbcolors[$order-1];
          if ($order > 9) {
            die "No more than 9 colors are available";
            print OUT "1 0 1 setrgbcolor\n"; # Magenta
          } 
          print OUT "stroke\n"; 
        }

        $j++;
        @prob = (0) x $numberLineage;  
        $binIndex = 0;
      }

      for (my $sourceI = 0; $sourceI < $numberLineage; $sourceI++)
      {
        my $v = $map->[$p][1 + $sourceI * $numberLineage + $destinationJ];
        die "Negative means no alignemnt in $p" if $v < 0;
        $prob[$sourceI] += ($v / ($clonaloriginsamplesize * $binSize));
      }
      $binIndex++;
    }
    # Postscript. 
    print OUT "newpath\n";
    print OUT "$xStart $yStart moveto\n"; 
    print OUT "$width $yStart lineto\n"; 
    print OUT "0 0 0 setrgbcolor\n";
    print OUT "stroke\n"; 
    my $tickIndex = 0;
    for (my $j = $xStart; $j < $width - $tickInterval; $j+=$tickInterval)
    {
      my $yTick = $yStart - $tickBar;
      my $yPos = $yStart - $tickBar - 15;
      print OUT "newpath\n";
      print OUT "$j $yStart moveto\n"; 
      print OUT "$j $yTick lineto\n"; 
      print OUT "0 0 0 setrgbcolor\n";
      print OUT "stroke\n"; 
      my $xLabel = $startPos + $tickIndex * $binSize * $tickInterval;
      print OUT "$j $yPos moveto ($xLabel) show\n";
      $tickIndex++;
    }
  }
  close OUT;
}

sub getLengthMap ($)
{
  my ($f) = @_;
  my $i = 0;
  open MAPLENGTH, $f or die "could not open $f $!";
  while (<MAPLENGTH>)
  {
    $i++;
  }
  close MAPLENGTH;
  return $i;
}

__END__
=head1 NAME

recombination-intensity1-probability.pl - Compute recombination intensity1 along a genome

=head1 VERSION

v1.0, Sun May 15 16:25:25 EDT 2011

=head1 SYNOPSIS

perl recombination-intensity1-probability.pl [-h] [-help] [-version] [-verbose]
  [-ri1map file] 
  [-ingene file] 
  [-out file] 

perl pl/recombination-intensity1-probability.pl ps 

perl pl/recombination-intensity1-probability.pl split -xmfa file.xmfa -ri1map rimap.txt

perl pl/recombination-intensity1-probability.pl wiggle

=head1 DESCRIPTION

command ps:

The number of recombination edge types at a nucleotide site along all of the
alignment blocks is computed for genes from an ingene file. 
Menu recombination-intensity1-map must be called first.

What we need includes:
1. recombination-intensity1-map file (-ri1map)
2. ingene file (-ingene)
3. out file (-out)

command split:

I split the rimap file to multiple files, each of which corresponds to a block
alignment. 

command wiggle:

UCSC genome browser container tracks can be used to display recombination
intensity.

=head1 OPTIONS

=over 8

=item B<-help> | B<-h>

Print the help message; ignore other arguments.

=item B<-man>

Print the full documentation; ignore other arguments.

=item B<-version>

Print program version; ignore other arguments.

=item B<-verbose>

Prints status and info messages during processing.

=item B<-ri1map> <file>

A recombination intensity 1 map file.

=item B<-ingene> <file>

An ingene file.

=item B<-out> <file>

An output file.

=item B<-pairs> <string>

  -pairs 0,3:0,4:1,3:1,4:3,0:3,1:4,0:4,1

=back

=head1 AUTHOR

Sang Chul Choi, C<< <goshng_at_yahoo_dot_co_dot_kr> >>

=head1 BUGS

If you find a bug please post a message mauve-analysis project at codaset dot
com repository so that I can make recombination-intensity1-probability.pl better.

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
