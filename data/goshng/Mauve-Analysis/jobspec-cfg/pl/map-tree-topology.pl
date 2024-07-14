#!/opt/local/bin/perl -w
#===============================================================================
#   Author: Sang Chul Choi, BSCB @ Cornell University, NY
#
#   File: map-tree-topology.pl
#   Date: Sat May  7 11:58:18 EDT 2011
#   Version: 1.0
#===============================================================================
use strict;
use warnings;
use Getopt::Long;
use Pod::Usage;

$| = 1; # Do not buffer output

my $VERSION = 'map-tree-topology.pl 1.0';

my $man = 0;
my $help = 0;
my %params = ('help' => \$help, 'h' => \$help, 'man' => \$man);        
GetOptions( \%params,
            'help|h',
            'man',
            'verbose',
            'version' => sub { print $VERSION."\n"; exit; },
            'ricombined=s',
            'ingene=s',
            'out=s',
            'treetopology=i' 
            ) or pod2usage(2);
pod2usage(1) if $help;
pod2usage(-exitstatus => 0, -verbose => 2) if $man;


require "pl/sub-simple-parser.pl";
require "pl/sub-newick-parser.pl";
require "pl/sub-ingene.pl";
require "pl/sub-array.pl";
sub get_genome_file ($$);
sub locate_block_in_genome ($$);
sub locate_nucleotide_in_block ($$);


#
################################################################################
## COMMANDLINE OPTION PROCESSING
################################################################################
#

my $ricombined;
my $ingene;
my $verbose = 0;
my $treetopology;
my $outfile;

if (exists $params{treetopology})
{
  $treetopology = $params{treetopology};
}
else
{
  &printError("you did not specify a treetopology directory");
}

if (exists $params{ricombined})
{
  $ricombined = $params{ricombined};
}
else
{
  &printError("you did not specify a ricombined directory");
}

if (exists $params{ingene})
{
  $ingene = $params{ingene};
}
else
{
  &printError("you did not specify an ingene file");
}

if (exists $params{verbose})
{
  $verbose = 1;
}

if (exists $params{out})
{
  open ($outfile, ">", $params{out}) or die "cannot open > $params{out} $!";
}
else
{
  $outfile = *STDOUT;   
}
################################################################################
## DATA PROCESSING
################################################################################

sub get_number_topology_change ($$$$);
sub map_tree_topology ($$$);
sub print_in_gene ($$$);

# my @genes = parse_in_gene ($ingene);
my @genes = maIngeneParseBlock ($ingene);
map_tree_topology (\@genes, $ricombined, $treetopology);
print_in_gene ($outfile, $ingene, \@genes);

if (exists $params{out})
{
  close $outfile;
}
sub map_tree_topology ($$$)
{
  my ($genes, $ricombined, $treetopology) = @_;
  my $numberGene = scalar @{ $genes };
  my $processedTime = 0;
  for (my $i = 0; $i < scalar @{ $genes }; $i++)
  {
    my $startTime = time; 
    my $h = $genes->[$i];
    # my $block = $h->{block};
    # my $start = $h->{blockstart};
    # my $end = $h->{blockend};
    my $block = $h->{blockidGene};
    my $start = $h->{blockStart};
    my $end = $h->{blockEnd};
    my $rifile = "$ricombined/$block";
    my ($mt,$mt2,$mt3,$mt4) = get_number_topology_change ($rifile, $start, $end, $treetopology);
    $h->{mt} = $mt;
    $h->{mt2} = $mt2;
    $h->{mt3} = $mt3;
    $h->{mt4} = $mt4;
    my $endTime = time; 
    my $elapsedTime = $endTime - $startTime;
    $processedTime += $elapsedTime;
    my $remainedGene = $numberGene - $i - 1;
    my $remainedTime = int(($processedTime/($i+1)) * $remainedGene / 60);
    if ($verbose == 1)
    {
      print STDERR "Genes: $h->{gene} ($i/$numberGene) $remainedTime min. to go\r";
    }
  }
}

sub get_number_topology_change ($$$$)
{
  my ($rifile, $start, $end, $treetopoogy) = @_;
  my $sampleSize = 0;
  my $v = 0;
  my $vTopology = 0;
  my $vCountTopology = 0;
  my $vTopologyChange = 0;
  open RI, $rifile or die "could not open $rifile";
  while (<RI>)
  {
    chomp;
    my @e = split /\t/;
    my @eGene = @e[$start .. $end]; 
    my %seen = (); my @uniquE = grep { ! $seen{$_} ++ } @eGene;
    my $numberUniqueTopology = scalar (@uniquE);
    $vCountTopology += $numberUniqueTopology;
    my $anyTopologyChange = 0;
    my $numberChangeTopology = 0;
    for (my $i = $start; $i < $end; $i++)
    {
      if ($e[$i] != $e[$i+1])
      {
        $numberChangeTopology++; 
        $anyTopologyChange++; 
      }
      if ($e[$i] != $treetopoogy)
      {
        $vTopology++;
      }
    }
    die "$numberUniqueTopology must be less than $numberChangeTopology by 2"
      unless $numberChangeTopology + 2 > $numberUniqueTopology;
    $v += $numberChangeTopology; 

    if ($e[$end] != $treetopoogy)
    {
      $vTopology++;
    }
    if ($anyTopologyChange > 0)
    {
      $vTopologyChange++; 
    }
    $sampleSize++;
  } 
  # $v $vCountTopology
  close RI;
  $v /= ($end - $start);
  $v /= $sampleSize;
  $vTopology /= (($end - $start + 1) * $sampleSize);
  $vCountTopology /= (($end - $start + 1) * $sampleSize);
  $vTopologyChange /= $sampleSize;
  return ($v, $vTopology, $vCountTopology, $vTopologyChange);
}

sub print_in_gene ($$$)
{
  my ($f, $ingene, $genes) = @_;
  
  # open INGENE, ">$ingene.temp" or die "$ingene.temp could be not opened";
  print $f "gene\tstart\tend\tstrand\tblockidGene\tblockStart\tblockEnd\t";
  print $f "geneStartInBlock\tgeneEndInBlock\tlenSeq\tgap\tmt\tmt2\tmt3\tmt4\n";
  for (my $i = 0; $i < scalar @{ $genes }; $i++)
  {
    my $rec = $genes->[$i];
    print $f "$rec->{gene}\t";
    print $f "$rec->{start}\t";
    print $f "$rec->{end}\t";
    print $f "$rec->{strand}\t";
    # blockidGene blockStart  blockEnd  geneStartInBlock  geneEndInBlock  lenSeq gap
    print $f "$rec->{blockidGene}\t";
    print $f "$rec->{blockStart}\t";
    print $f "$rec->{blockEnd}\t";
    print $f "$rec->{geneStartInBlock}\t";
    print $f "$rec->{geneEndInBlock}\t";
    print $f "$rec->{lenSeq}\t";
    print $f "$rec->{lenSeq}\t";
    # print $f "$rec->{block}\t";
    # print $f "$rec->{blockstart}\t";
    # print $f "$rec->{blockend}\t";
    # print $f "$rec->{genelength}\t";
    # print $f "$rec->{proportiongap}\t";
    print $f "$rec->{mt}\t";
    print $f "$rec->{mt2}\t";
    print $f "$rec->{mt3}\t";
    print $f "$rec->{mt4}\n";
  }
  #close INGENE;
  #rename "$ingene.temp", $ingene
}

sub printError {
    my $msg = shift;
    print STDERR "ERROR: ".$msg.".\n\nTry \'-h\' option for more information.\nExit program.\n";
    exit(0);
}
__END__
=head1 NAME

map-tree-topology.pl - Measure number of topology changes

=head1 VERSION

map-tree-topology.pl 1.0

=head1 SYNOPSIS

perl map-tree-topology.pl [-h] [-help] [-version] [-verbose]
  [-ricombined directory] 
  [-ingene file] 

=head1 DESCRIPTION

Number of tree topology changes are computed for all of the genes in ingene
file.

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

=item B<***** INPUT OPTIONS *****>

=item B<-ricombined> <directory>

A directory that contains the ri-REPLICATE-combined.

=item B<-infile> <file>

An infile from locate-gene-in-block menu.

=back

=head1 AUTHOR

Sang Chul Choi, C<< <goshng_at_yahoo_dot_co_dot_kr> >>

=head1 BUGS

If you find a bug please post a message mauve-analysis project at codaset dot
com repository so that I can make map-tree-topology.pl better.

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
