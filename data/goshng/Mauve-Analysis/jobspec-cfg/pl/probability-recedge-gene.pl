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
#===============================================================================
#   Author: Sang Chul Choi, BSCB @ Cornell University, NY
#
#   File: probability-recedge-gene.pl
#   Date: Wed Jun  8 15:37:24 EDT 2011
#   Version: 1.0
#===============================================================================
use strict;
use warnings;
use Getopt::Long;
use Pod::Usage;

$| = 1; # Do not buffer output

my $VERSION = 'probability-recedge-gene.pl 1.0';

my $man = 0;
my $help = 0;
my %params = ('help' => \$help, 'h' => \$help, 'man' => \$man);        
GetOptions( \%params,
            'help|h',
            'man',
            'verbose',
            'version' => sub { print $VERSION."\n"; exit; },
            'ri1map=s',
            'genbank=s',
            'clonaloriginsamplesize=i',
            'out=s',
            'latex'
            ) or pod2usage(2);
pod2usage(1) if $help;
pod2usage(-exitstatus => 0, -verbose => 2) if $man;
sub getLengthMap ($);

require "pl/sub-ingene.pl";
require "pl/sub-error.pl";
require "pl/sub-array.pl";

# Delete these if not needed.
require "pl/sub-simple-parser.pl";
require "pl/sub-newick-parser.pl";
require "pl/sub-xmfa.pl";

################################################################################
## COMMANDLINE OPTION PROCESSING
################################################################################

my $ri1map;
my $ingene;
my $genbank;
my $out;
my $outfile;
my $clonaloriginsamplesize;
my $pairs;
my $verbose = 0;
my $latex = 0;

if (exists $params{ri1map})
{
  $ri1map = $params{ri1map};
}
else
{
  &printError("you did not specify an ri1map file that contains recombination intensity 1 measure");
}

if (exists $params{out})
{
  $out = $params{out};
  open ($outfile, ">", $params{out}) or die "cannot open > $params{out}: $!";
}
else
{
  $outfile = *STDOUT;   
}

if (exists $params{genbank})
{
  $genbank = $params{genbank};
}
else
{
  &printError("you did not specify a genbank file");
}

if (exists $params{clonaloriginsamplesize})
{
  $clonaloriginsamplesize = $params{clonaloriginsamplesize};
}
else
{
  &printError("you did not specify an clonaloriginsamplesize");
}

if (exists $params{verbose})
{
  $verbose = 1;
}

if (exists $params{latex})
{
  $latex = 1;
}

################################################################################
## DATA PROCESSING
################################################################################
my $itercount = 0;
my $blockLength;
my $blockidForProgress;
my $numberTaxa = 5;
my $numberLineage = 2 * $numberTaxa - 1;

################################################################################
# Find coordinates of the reference genome.
################################################################################
sub parse_genbank ($);
sub genbankOpen ($);
sub genbankNext ($);
sub genbankClose ($);
sub riOpen ($);
sub riCompute ($$$$);

if ($latex == 1)
{
  print $outfile "\\documentclass{article}\n\\begin{document}\n";
  print $outfile "\\begin{tabular}{ | l l l l l l l l l l | }\n";
  print $outfile "\\hline\n";
  print $outfile "Locus & srceI & destI & fragPortion & fragStart & fragEnd & coverage & gene start & gene end & gene product\\\\\n";
  print $outfile "\\hline\n";
}

my @notProcessedGene;
open (my $riFile, $ri1map) or die $!;
my $pos = 1;

my $genbankFile = genbankOpen ($genbank);
my %gene = genbankNext ($genbankFile);
while ($gene{locus} ne "")
{
  # print "$gene{locus}\t$gene{start}\t$gene{end}\t$gene{product}\n";
  if ($pos <= $gene{start})
  {
    # my ($srceI, $destI, $max, $coverage);
    # ($pos, $srceI, $destI, $max, $coverage) = riCompute ($riFile, $pos, $gene{start}, $gene{end});
    my ($coverage, $reTransfer);
    ($pos, $coverage, $reTransfer) = riCompute ($riFile, $pos, $gene{start}, $gene{end});
    for my $h (@{ $reTransfer })
    {
      my $index = $h->{index};
      my $srceI = int($index / 9);
      my $destI = $index % 9;
      my $fragStart = $h->{start};
      my $fragEnd = $h->{end};
      my $fragPortion = int(($fragEnd - $fragStart) / ($gene{end} - $gene{start}) * 100);
      if ($latex == 0)
      {
        print $outfile "$gene{locus}\t$srceI\t$destI\t$fragPortion\t$fragStart\t$fragEnd\t$coverage\t$gene{start}\t$gene{end}\t$gene{product}\n";
      }
      else
      {
        my $locusname = $gene{locus};
        $locusname =~ s/_/\\_/g; 
        print $outfile "$locusname & $srceI & $destI & $fragPortion & $fragStart & $fragEnd & $coverage & $gene{start} & $gene{end} & $gene{product} \\\\\n";
      }
    }
  }
  else
  {
    # Genes that already started.
    # Compute posterior probability for these genes later.
    # print STDERR "NOT PROCESSED $gene{locus}\t$gene{start}\t$gene{end}\t$gene{product}\n";
    push @notProcessedGene, { %gene };
  }
  %gene = genbankNext ($genbankFile);
}
close ($genbankFile);
close ($riFile);

#################################################
# Compute posterior probability for the unprocessed genes. 
open ($riFile, $ri1map) or die $!;
$pos = 1;
for my $g ( @notProcessedGene ) {
  my ($coverage, $reTransfer);
  ($pos, $coverage, $reTransfer) = riCompute ($riFile, $pos, $g->{start}, $g->{end});
  for my $h (@{ $reTransfer })
  {
    my $index = $h->{index};
    my $srceI = int($index / 9);
    my $destI = $index % 9;
    my $fragStart = $h->{start};
    my $fragEnd = $h->{end};
    my $fragPortion = int(($fragEnd - $fragStart) / ($g->{end} - $g->{start}) * 100);
    if ($latex == 0)
    {
      print $outfile "$g->{locus}\t$srceI\t$destI\t$fragPortion\t$fragStart\t$fragEnd\t$coverage\t$g->{start}\t$g->{end}\t$g->{product}\n";
    }
    else
    {
      my $locusname = $g->{locus};
      $locusname =~ s/_/\\_/g; 
      print $outfile "$locusname & $srceI & $destI & $fragPortion & $fragStart & $fragEnd & $coverage & $g->{start} & $g->{end} & $g->{product}\\\\\n";
    }
  }
}
close ($riFile);

if ($latex == 1)
{
  print $outfile "\\end{tabular}\n";
  print $outfile "\\end{document}\n";
}

my $lengthGenome = 0;
my @genes;

exit;

################################################################################
## END OF DATA PROCESSING
################################################################################

################################################################################
## FUNCTION DEFINITION
################################################################################
sub riCompute ($$$$)
{
  my ($f, $pos, $start, $end) = @_;
  my $threshould = 0.9 * $clonaloriginsamplesize;
  my @reTransfer;
  my $line;

  die "$pos is greater than $start" if $pos > $start;
  while ($pos < $start)
  {
    $line = <$f>;
    $pos++;
  }

  my @m = (0) x 81;
  my $numberOfSitesOfNonzero = 0;
  while ($pos <= $end)
  {
    $line = <$f>;
    my @e = split /\t/, $line;
    if ($e[1] < 0)
    {
      # no map.
    }
    else
    {
      $numberOfSitesOfNonzero++; 
      for (my $i = 0; $i < 72; $i++)
      {
        die "negative values $pos $i" if $e[$i+1] < 0;
        $m[$i] += $e[$i+1];
        if ($e[$i+1] > $threshould)
        {
          my $found = 0;
          for my $h (@reTransfer) 
          {
            if ($h->{index} == $i and $h->{end} == $pos - 1)
            {
              $h->{end} = $pos;
              $found = 1;
              last;
            }
          }
          if ($found == 0)
          {
            my $rec = {};
            $rec->{index} = $i;
            $rec->{start} = $pos;
            $rec->{end} = $pos;
            push @reTransfer, $rec;
          }
        }
      }
    }
    die "Incorrect position" unless $e[0] == $pos;
    $pos++;
  }

  my $coverage = int($numberOfSitesOfNonzero / ($end - $start + 1) * 100);

  my $srceI = -1;
  my $destI = -1;
  my $max = -1;

  if ($numberOfSitesOfNonzero > 0)
  {
    for (my $i = 0; $i < 81; $i++)
    {
      $m[$i] /= ($numberOfSitesOfNonzero * $clonaloriginsamplesize);
    }

    # Find the max and its index.
    my $maxIndex;
    for (my $i = 0; $i < 81; $i++)
    {
      if ($m[$i] > $max)
      {
        $max = $m[$i];
        $maxIndex = $i;
      }
    }
    $srceI = int($maxIndex / 9);
    $destI = $maxIndex % 9;

  }
  else
  {
    # No map.
  }

  return ($pos, $coverage, \@reTransfer);
}

sub genbankOpen ($)
{
  my ($f) = @_;
  open GENBANK, $f or die "Could not open $f $!";
  return \*GENBANK;
}

sub genbankNext ($)
{
  my ($f) = @_;
  my %gene;
  $gene{locus} = "";

  my $foundGene = 0;
  my $foundLocus = 0;
  my $foundPosition = 0;
  while (my $line = <$f>)
  {
    chomp $line;
    if ($line =~ /^\s+gene\s+/)
    {
      $foundGene = 1;
    }
    if ($foundGene == 1)
    {
      if ($line =~ /^\s+\/locus_tag=\"(\w+)\"/)
      {
        $gene{locus} = $1;
        $foundLocus = 1;
      }
    }

    if ($foundLocus == 1)
    {
      if ($line =~ /^\s+CDS\s+/
          or $line =~ /^\s+rRNA\s+/
          or $line =~ /^\s+tRNA\s+/)
      {
        if ($line =~ /(\d+)\.\.(\d+)/)
        {
          $gene{start} = $1;
          $gene{end} = $2;
          $foundPosition = 1;
        }
      }
    }

    if ($foundPosition == 1)
    {
      if ($line =~ /^\s+\/product=\"/)
      {
        my $product = "";
        if ($line =~ /\"(.+)\"/)
        {
          $product = $1;
        }
        elsif ($line =~ /\"(.+)/)
        {
          $product = $1;
          $line = <$f>;
          while ($line !~ /^\s+\//)
          {
            chomp $line;
            $product .= $line;
            $line = <$f>;
          }
          $product =~ s/\s+/ /g;
          $product =~ s/\"//g;
        }
        $gene{product} = $product;
        last;
      }
    }

  }
  return %gene;
}

sub genbankClose ($)
{
  my ($f) = @_;
  close $f;
}

sub parse_genbank ($)
{
  my ($f) = @_;
  open GENBANK, $f or die "Could not open $f $!";
  while (<GENBANK>)
  {
    
  }
  close GENBANK;
}

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

__END__
=head1 NAME

probability-recedge-gene.pl - Compute probability of recombination of genes

=head1 VERSION

v1.0, Sun May 15 16:25:25 EDT 2011

=head1 SYNOPSIS

perl probability-recedge-gene.pl [-h] [-help] [-version] [-verbose]
  [-ri1map file] 
  [-ingene file] 
  [-genbank file] 
  [-out file] 
  [-latex] 

perl pl/probability-recedge-gene.pl \
     -ri1map ri1-refgenome4-map.txt \
     -clonaloriginsamplesize 1001 \
     -genbank file.gbk



=head1 DESCRIPTION

The number of recombination edge types at a nucleotide site along all of the
alignment blocks is computed for genes from an ingene file. 
Menu recombination-intensity1-map must be called first.

What we need includes:
1. recombination-intensity1-map file (-ri1map)
2. ingene file (-ingene)
3. out file (-out)

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

=item B<-genbank> <file>

A GenBank file.

=item B<-ingene> <file>

An ingene file.

=item B<-out> <file>

An output file.

=item B<-latex>

The output file in LaTeX.


=item B<-pairs> <string>

  -pairs 0,3:0,4:1,3:1,4:3,0:3,1:4,0:4,1

=back

=head1 AUTHOR

Sang Chul Choi, C<< <goshng_at_yahoo_dot_co_dot_kr> >>

=head1 BUGS

If you find a bug please post a message mauve-analysis project at codaset dot
com repository so that I can make probability-recedge-gene.pl better.

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
