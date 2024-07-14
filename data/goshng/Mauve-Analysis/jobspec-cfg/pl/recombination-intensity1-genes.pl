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
#   File: recombination-intensity1-genes.pl
#   Date: Tue May 17 14:34:30 EDT 2011
#   Version: 1.0
#===============================================================================
use strict;
use warnings;
use Getopt::Long;
use Pod::Usage;

$| = 1; # Do not buffer output

my $VERSION = 'recombination-intensity1-genes.pl 1.0';

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
            'refgenome=i',
            'ri1map=s',
            'ingene=s',
            'clonaloriginsamplesize=i',
            'pairs=s',
            'numberSpecies=i',
            'out=s',
            'pairm=s',
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

# Delete these if not needed.
require "pl/sub-simple-parser.pl";

################################################################################
## COMMANDLINE OPTION PROCESSING
################################################################################

my $ri1map;
my $ingene;
my $out;
my $outfile;
my $clonaloriginsamplesize;
my $pairs;
my $pairm = "all";
my $verbose = 0;
my $xml;
my $xmfa;
my $refgenome; 

if (exists $params{xml})
{
  $xml = $params{xml};
}
else
{
  &printError("you did not specify an xml directory that contains Clonal Origin 2nd run results");
}

if (exists $params{xmfa})
{
  $xmfa = $params{xmfa};
}
else
{
  &printError("you did not specify a core genome alignment");
}

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
  open ($outfile, ">", $out) or die "cannot open > $out: $!";
}
else
{
  $outfile = *STDOUT;   
}

if (exists $params{ingene})
{
  $ingene = $params{ingene};
}
else
{
  &printError("you did not specify an ingene file");
}

if (exists $params{clonaloriginsamplesize})
{
  $clonaloriginsamplesize = $params{clonaloriginsamplesize};
}
else
{
  &printError("you did not specify an clonaloriginsamplesize");
}

if (exists $params{refgenome})
{
  $refgenome = $params{refgenome};
}
else
{
  $refgenome = -1;
}

if (exists $params{pairs})
{
  $pairs = $params{pairs};
}

if (exists $params{pairm})
{
  $pairm = $params{pairm};
  unless ($pairm eq 'all' or 
          $pairm eq 'topology' or 
          $pairm eq 'notopology' or 
          $pairm eq 'pair')
  {
    die "-pairm options must be one of all, topology, notopology, and pair";
  }
}

if (exists $params{verbose})
{
  $verbose = 1;
}

################################################################################
## DATA PROCESSING
################################################################################

my $progressbar = "Block: 0 Not Yet Determined min to go";

my $tag;
my $content;
my %recedge;
my $itercount = 0;
my $blockLength;
my $blockidForProgress;
my $speciesTree = get_species_tree ("$xml.1");
my $numberTaxa = get_number_leave ($speciesTree);
my $tree = maNewickParseTree ($speciesTree);
# maNewickPrintTree ($tree);
my $numberLineage = 2 * $numberTaxa - 1;

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
  my @genes = maIngeneParseBlock ($ingene); 
  maIngenePrintBlock ("a", \@genes);

  my $pairM; 
  if ($pairm eq 'all')
  {
    $pairM = maNewicktFindRedEdge ($tree);
  }
  elsif ($pairm eq 'topology')
  {
    $pairM = maNewicktFindRedEdgeChangeTopology ($tree);
  }
  elsif ($pairm eq 'notopology')
  {
    $pairM = maNewicktFindRedEdgeNotChangeTopology ($tree);
  }
  elsif ($pairm eq 'pair')
  {
    $pairM = maNewicktFindRedEdgePair ($tree, $pairs);
  }
  else
  {
    die "-pairm options must be one of all, topology, notopology, and pair";
  }
  my @blockStart = maXmfaGetBlockStart ($xmfa);
  maRiGetGenes (\@genes, $ri1map, \@blockStart, $pairM, $numberLineage, $clonaloriginsamplesize);
  # maRiGetGenesUsingRimapDirectory (\@genes, $ri1map, \@blockStart, $pairM, $numberLineage, $clonaloriginsamplesize);
  maIngenePrintBlockRi ($outfile, \@genes);
}
else
{
  my @genes = parse_in_gene ($ingene); 

  my @pairSourceDestination = getPairs ($pairs);
  #for my $i ( 0 .. $#pairSourceDestination) {
    #print "\t [ @{$pairSourceDestination[$i]} ],\n";
  #}

  #test (\@genes, $ri1map, \@pairSourceDestination);
  getRI1Genes (\@genes, $ri1map, \@pairSourceDestination);
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
  open OUT, ">$out" or die "Could not open $out $!";
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

recombination-intensity1-genes.pl - Compute recombination intensity on genes

=head1 VERSION

v1.0, Sun May 15 16:25:25 EDT 2011

=head1 SYNOPSIS

perl recombination-intensity1-genes.pl [-h] [-help] [-version] [-verbose]
  [rimap]
  [-ri1map file] 
  [-ingene file] 
  [-clonaloriginsamplesize number]
  [-pairs string] 
  [-numberSpecies number]
  [-out file] 

=head1 DESCRIPTION

The number of recombination edge types at a nucleotide site along all of the
alignment blocks is computed for genes from an ingene file. 
Menu recombination-intensity1-map must be called first.

What we need includes:
1. recombination-intensity1-map file (-ri1map)
2. ingene file (-ingene)
3. out file (-out)
4. sample size (-clonaloriginsamplesize) 
5. number of species (-numberSpecies number)

=head1 OPTIONS

=over 8

=item B<rimap>

Use rimap. 

=item B<-help> | B<-h>

Print the help message; ignore other arguments.

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

=item B<-clonaloriginsamplesize> <number>

The sample size of recombinant trees.

=item B<-numberSpecies> <number>

The number of species.

=item B<-pairs> <string>

  -pairs 0,3:0,4:1,3:1,4:3,0:3,1:4,0:4,1

=item B<-pairm> <string>

The string can be all, topology, notopology, pair.

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


=back

=head1 AUTHOR

Sang Chul Choi, C<< <goshng_at_yahoo_dot_co_dot_kr> >>

=head1 BUGS

If you find a bug please post a message mauve-analysis project at codaset dot
com repository so that I can make recombination-intensity1-genes.pl better.

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

