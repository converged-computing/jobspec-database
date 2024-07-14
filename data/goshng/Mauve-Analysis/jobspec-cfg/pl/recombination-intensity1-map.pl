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
use XML::Parser;
use Getopt::Long;
use Pod::Usage;
require "pl/sub-simple-parser.pl";
require "pl/sub-newick-parser.pl";
require "pl/sub-error.pl";
require "pl/sub-array.pl";
require "pl/sub-xmfa.pl";
require "pl/sub-gbk.pl";

$| = 1; # Do not buffer output
my $VERSION = 'recombination-intensity1-map.pl 1.0';

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
            'gbk=s',
            'refgenome=i',
            'blockid=i',
            'refgenomelength=i',
            'numberblock=i',
            'out=s',
            '<>' => \&process
            ) or pod2usage(2);
pod2usage(1) if $help;
pod2usage(-exitstatus => 0, -verbose => 2) if $man;

################################################################################
## COMMANDLINE OPTION PROCESSING
################################################################################

my $xml;
my $xmfa;
my $refgenome; 
my $refgenomelength;
my $numberblock; 
my $verbose = 0;
my $out;
my $outfile;

if (exists $params{out})
{
  $out = $params{out};
  open ($outfile, ">", $out) or die "cannot open > $out: $!";
}
else
{
  $outfile = *STDOUT;   
}

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

# Use refgenome being -1 for finding whether I use a reference genome or not.
if (exists $params{refgenome})
{
  $refgenome = $params{refgenome};
}
else
{
  $refgenome = -1;
}

unless ($refgenome == -1)
{
  if ($cmd eq "block")
  {
    die "Command block must not be used with option -refgenome"; 
  }

  if (exists $params{refgenomelength})
  {
    $refgenomelength = $params{refgenomelength};
  }
  else
  {
    if (exists $params{gbk})
    {
      $refgenomelength = peachGbkLength ($params{gbk});
    }
    else
    {
      &printError("you did not specify the length of $refgenome-th reference genome");
    }
  }
}

if (exists $params{numberblock})
{
  $numberblock = $params{numberblock};
}
else
{
  &printError("you did not specify the number of blocks");
}

if ($cmd eq "block")
{
  unless (exists $params{blockid})
  {
    &printError("Command $cmd requires -blockid");
  }
}

if (exists $params{verbose})
{
  $verbose = 1;
}

################################################################################
## DATA PROCESSING
################################################################################

my $progressbar = "Block: 0/$numberblock Not Yet Determined min to go";
my $tag;
my $content;
my %recedge;
my $itercount = 0;
my $blockLength;
my $blockidForProgress;
my $speciesTree = get_species_tree ("$xml.1");
my $numberTaxa = get_number_leave ($speciesTree);
my $numberLineage = 2 * $numberTaxa - 1;
my $startBlock;
my $endBlock;

################################################################################
# Find coordinates of the reference genome.
################################################################################

my @blockLocationGenome;
$startBlock = 1;
$endBlock = $numberblock;
if ($refgenome == -1)
{
  @blockLocationGenome = getBlockSizeConfiguration ($xmfa, $numberblock);
  if ($cmd eq "block")
  {
    # 1st Block size: 100
    # start: 1
    # end: 100
    # 1st Block size: 200
    # start: 101
    # end: 300
    $refgenomelength = $blockLocationGenome[$params{blockid}-1]->{end}
                       - $blockLocationGenome[$params{blockid}-1]->{start} + 1;
    $startBlock = $params{blockid};
    $endBlock = $params{blockid};
  }
  else
  {
    $refgenomelength = $blockLocationGenome[$#blockLocationGenome]->{end};
  }
}
else
{
  @blockLocationGenome = getBlockConfiguration ($refgenome, $xmfa, $numberblock);
}

if ($verbose == 1)
{
  for my $i (0 .. $#blockLocationGenome)
  {
    my $href = $blockLocationGenome[$i];
    print STDERR "$i:$href->{start} - $href->{end}\n"; 
  }
  print STDERR "Total Length: $refgenomelength\n";
}

# mapImport index is 1-based, and blockImmport index is 0-based.
# map starts at 1.
# block starts at 0.
# mapImport is created.
my @blockImport;
my @mapBlockImport;

if ($verbose == 1)
{
  print STDERR "A new mapImport is being constructed ... ";
}
my @mapImport = create3DMatrix ($numberLineage, 
                                $numberLineage, 
                                $refgenomelength + 1, -1);
if ($verbose == 1)
{
  print STDERR "done!\n";
}

# mapImport is filled in.
my $offsetPosition = 0;
my $prevBlockLength = 1;
my $processedTime = 0;
# for my $blockID (1 .. $numberblock)
for my $blockID ($startBlock .. $endBlock)
{
  # last if $blockID == 2;
  my $startTime = time; 
  $blockidForProgress = $blockID;
  my $f = "$xml.$blockID";
  $blockLength = get_block_length ($f);
  my $blockSize = xmfaBlockSize ("$xmfa.$blockID");
  die "$blockLength and $blockSize are different" 
    unless $blockLength == $blockSize;
  my $href = $blockLocationGenome[$blockID-1];
  die "$href->{start} < $href->{end}" 
    unless $href->{start} < $href->{end};
  $offsetPosition = $href->{start};

  @mapBlockImport = create3DMatrix ($numberLineage, 
                                    $numberLineage, 
                                    $blockLength, 0);
  my $parser = new XML::Parser();
  $parser->setHandlers(Start => \&startElement,
                       End => \&endElement,
                       Char => \&characterData,
                       Default => \&default);

  $itercount = 0;
  my $doc;
  eval{ $doc = $parser->parsefile($f)};
  die "Unable to parse XML of $f, error $@\n" if $@;

  # Put the block map to the output.
  my @sites;
  my ($b, $e, $strand);
  if ($refgenome == -1)
  {
    @sites = ('a') x $blockSize;
    if ($cmd eq "block")
    {
      $b = 1;
      $e = $href->{end} - $href->{start} + 1;
    }
    else
    {
      $b = $href->{start};
      $e = $href->{end};
    }
    $strand = '+';
  }
  else
  {
    @sites = locateNucleotideBlock ("$xmfa.$blockID", $refgenome);
    ($b, $e, $strand) = locateBlockGenome ("$xmfa.$blockID", $refgenome);
  }
  die "$b < $e in $xmfa.$blockID" unless $b < $e;

  if ($strand eq '-')
  {
    my @sites_reversed = reverse (@sites);
    @sites = @sites_reversed;
  }

  my $c = 0; 
  for my $i ( 0 .. $#sites ) 
  {
    next if $sites[$i] eq '-';
    my $pos = $b + $c;
    for (my $j = 0; $j < $numberLineage; $j++)
    {
      for (my $k = 0; $k < $numberLineage; $k++)
      {
        $mapImport[$j][$k][$pos] = $mapBlockImport[$j][$k][$i];
      }
    }
    $c++;
  }
  $c--;
  die "$e and $b + $c do not match" unless $e == $b + $c;

  if ($verbose == 1)
  {
    my $endTime = time; 
    my $elapsedTime = $endTime - $startTime;
    $processedTime += $elapsedTime;
    my $remainedBlock= $numberblock - $blockID;
    my $remainedTime = int(($processedTime/$blockID) * $remainedBlock / 60);
    $progressbar = "Block: $blockID/$numberblock $remainedTime min to go";
  }
}

# mapImport is printed out.
for (my $i = 1; $i <= $refgenomelength; $i++)
{
  my $pos;
  if ($cmd eq "block")
  {
    $pos = $blockLocationGenome[$params{blockid}-1]->{start} + $i - 1;
  }
  else
  {
    $pos = $i;
  }
  print $outfile "$pos";
  for (my $j = 0; $j < $numberLineage; $j++)
  {
    for (my $k = 0; $k < $numberLineage; $k++)
    {
      print $outfile "\t", $mapImport[$j][$k][$i];
    }
  }
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

sub startElement {
  my( $parseinst, $element, %attrs ) = @_;
  $tag = $element;
  SWITCH: {
    if ($element eq "Iteration") 
    {
      $itercount++;
      @blockImport = create3DMatrix ($numberLineage, 
                                     $numberLineage, 
                                     $blockLength, 0);
      last SWITCH;
    }
  }
}

sub endElement {
  my ($p, $elt) = @_;

  if ($elt eq "efrom") {
    $recedge{efrom} = $content;
  }
  if ($elt eq "eto") {
    $recedge{eto} = $content;
  }
  if ($elt eq "start") {
    $recedge{start} = $content;
  }
  if ($elt eq "end") {
    $recedge{end} = $content;
  }

  if ($elt eq "recedge")
  {
    for (my $i = $recedge{start}; $i < $recedge{end}; $i++)
    {
      # NOTE: efrom -> eto. This used to be eto -> efrom.
      $blockImport[$recedge{efrom}][$recedge{eto}][$i]++;
    }
  }
  if ($elt eq "Iteration")
  {
    for (my $i = 0; $i < $blockLength; $i++)
    {
      my $pos = $i;
      for (my $j = 0; $j < $numberLineage; $j++)
      {
        for (my $k = 0; $k < $numberLineage; $k++)
        {
          if ($blockImport[$j][$k][$pos] > 0) 
          {
            $blockImport[$j][$k][$pos] = 1;
          }
        }
      }
    }

    for (my $i = 0; $i < $blockLength; $i++)
    {
      my $pos = $i;
      for (my $j = 0; $j < $numberLineage; $j++)
      {
        for (my $k = 0; $k < $numberLineage; $k++)
        {
          $mapBlockImport[$j][$k][$pos] += $blockImport[$j][$k][$i];
        }
      }
    }
    if ($verbose == 1)
    {
      print STDERR "$progressbar - Iteration $itercount\r";
    }
  }
  $tag = "";
  $content = "";
}

sub characterData {
  my( $parseinst, $data ) = @_;
  $data =~ s/\n|\t//g;
  $content .= $data;
}

sub default {
}
__END__
=head1 NAME

recombination-intensity1-map.pl - Compute recombination intensity1 along a genome

=head1 VERSION

v1.0, Sun May 15 16:25:25 EDT 2011

=head1 SYNOPSIS

perl recombination-intensity1-map.pl [-h] [-help] [-version] [-verbose]
  [-xml file base name] 
  [-xmfa file base name] 
  [-refgenome number] 
  [-refgenomelength number] 
  [-numberblock number] 
  [-out file] 

perl pl/recombination-intensity1-map.pl -xml core_co.phase3.xml
     -xmfa core_alignment.xmfa \
     -numberblock 274 \
     -out outfile

perl pl/recombination-intensity1-map.pl -xml core_co.phase3.xml
     -xml core_co.phase3.xml \
     -xmfa core_alignment.xmfa \
     -refgenome 4 \
     -refgenomelength $REFGENOMELENGTH \
     -numberblock 274 \
     -out outfile

perl pl/recombination-intensity1-map.pl -xml core_co.phase3.xml
     -xml core_co.phase3.xml \
     -xmfa core_alignment.xmfa \
     -refgenome 4 \
     -numberblock 274 \
     -out outfile

=head1 DESCRIPTION

The number of recombination edge types at a nucleotide site along all of the
alignment blocks is computed.

What we need includes:
1. ClonalOrigin output2 (-xml)
2. Genomes alignment (-xmfa)
3. Reference genome ID (-refgenome)

recombination-intensity1-map.pl help you compute recombinant edge counts
along a genome. I order all the alignment blocks with respect to
one of genomes. I would use the first genome in the alignment. I
need to use the species tree that is in the clonal origin output
files. Note that the numbering of internal nodes in the input
species tree and that of the clonal output files were different. I
have to use the species tree in the clonal origin to locate
internal nodes. Using the species tree I should be able to find the
species and their ancestors. Find which ordered pairs are possible
and which others are not. I need to parse the species tree in a
clonal origin outputfile. 
Consider a species tree with recombinant edges: e.g., Didelot's
2010 ClonalOrigin paper. For each site of an alignment block I can
have a matrix where element is a binary character. A site is
affected by multiple recombinant edges. It is possible that
recombinant edges with the same arrival and departure affect a
single site. It happened in the ClonalOrigin output file. If you simply
count recombinant edges, you could count some recombinant edge type
two or more times. To avoid the multiple count we use a matrix with
binary values. Then, we sum the binary matrices across all the
iteratons.
Note that the source and destination edges could be reversed. Be
careful not to reverse it. I used to use to -> from not from -> to.
Now, I use from -> to for each position.

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

=item B<-xml> <file base name>

A base name for ClonalOrigin XML files
that contains the 2nd phase run result from Clonal Origin. A number of XML files
for blocks are produced by appending dot and a block ID.

  -xml $RUNCLONALORIGIN/output2/${REPLICATE}/core_co.phase3.xml

The XML file for block ID of 1 is

  -xml $RUNCLONALORIGIN/output2/${REPLICATE}/core_co.phase3.xml.1

=item B<-xmfa> <file base name>

A base name for xmfa formatted alignment blocks. Each block alignment is
produced by appending a dot and block ID.

  -xmfa $DATADIR/core_alignment.xmfa

A XMFA block alignment for block ID of 1 is

  -xmfa $DATADIR/core_alignment.xmfa.1

=item B<-refgenome> <number>

A reference genome ID. If no reference genome is given by users, I use only
blocks to compute maps.

=item B<-refgenomelength> <number>

The length of the reference genome. If refgenome is given, its length must be
given.

=item B<-numberblock> <number>

The number of blocks.

=item B<-out> <file>

If this is given, all of the output is written to the file. Otherwise, standard
output is used.

=back

=head1 AUTHOR

Sang Chul Choi, C<< <goshng_at_yahoo_dot_co_dot_kr> >>

=head1 BUGS

If you find a bug please post a message mauve-analysis project at codaset dot
com repository so that I can make recombination-intensity1-map.pl better.

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
