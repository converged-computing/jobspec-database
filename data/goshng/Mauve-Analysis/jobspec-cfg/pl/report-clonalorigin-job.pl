#!/opt/local/bin/perl -w
#===============================================================================
#   Author: Sang Chul Choi, BSCB @ Cornell University, NY
#
#   File: report-clonalorigin-job.pl
#   Date: Thu Apr 21 12:42:10 EDT 2011
#   Version: 1.0
#
#   Usage:
#      perl report-clonalorigin-job.pl [options]
#
#      Try 'perl report-clonalorigin-job.pl -h' for more information.
#
#   Purpose: A set of clonal origin output files are checked.
#
#   Note that I started to code this based on PRINSEQ by Robert SCHMIEDER at
#   Computational Science Research Center @ SDSU, CA as a template. Some of
#   words are his not mine, and credit should be given to him. 
#===============================================================================

use strict;
use warnings;
use File::Basename;
use XML::Parser;
use Getopt::Long;
use Pod::Usage;
use File::Temp qw(tempfile);
require "pl/sub-error.pl";

$| = 1; # Do not buffer output

my $VERSION = 'report-clonalorigin-job.pl 1.0';

my $man = 0;
my $help = 0;
my %params = ('help' => \$help, 'h' => \$help, 'man' => \$man);        
GetOptions( \%params,
            'help|h',
            'man',
            'verbose',
            'version' => sub { print $VERSION."\n"; exit; },
            'samplesize=i',
            'xmlbase=s',
            'database=s'
            ) or pod2usage(2);
pod2usage(1) if $help;
pod2usage(-exitstatus => 0, -verbose => 2) if $man;

=head1 NAME

report-clonalorigin-job.pl - Build a heat map of recombination.

=head1 VERSION

report-clonalorigin-job.pl 0.1.0

=head1 SYNOPSIS

perl report-clonalorigin-job.pl [-h] [-help] [-version] 
  [-xml xmlfile] 

perl report-clonalorigin-job.pl -xmlbase dir/core_co.phase3.xml \
  -database dir2/core_alignment.xmfa \
  -samplesize 1001

=head1 DESCRIPTION

A set of XML Clonal Origin files is checked if they conform to the format of
a clonal origin output file.

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

=item B<-xmlbase> <xmlfile base name>

A clonal origin XML file base name.

=back

=head1 AUTHOR

Sang Chul Choi, C<< <goshng_at_yahoo_dot_co_dot_kr> >>

=head1 BUGS

If you find a bug please post a message rnaseq_analysis project at codaset dot
com repository so that I can make report-clonalorigin-job.pl better.

=head1 COPYRIGHT

Copyright (C) 2011  Sang Chul Choi

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

require "pl/sub-simple-parser.pl";

my $xmlFilebase;
my $xmfaFilebase;

if (exists $params{xmlbase})
{
  $xmlFilebase = $params{xmlbase};
}
else
{
  &printError("you did not specify an XML file base name");
}

if (exists $params{database})
{
  $xmfaFilebase = $params{database};
}
else
{
  &printError("you did not specify an XMFA file base name");
}

#
################################################################################
## DATA PROCESSING
################################################################################
#

sub print_xml_one_line ($$);

##############################################################
# Global variables
##############################################################
my $delta;
my $rho;
my $Lt;           # Total length of all of the blocks
my $numberBlock; 
my @blockLength;
my @blocks;
my $tag;
my $content;
my %recedge;
my $itercount=0;

my @xmfaFiles = <$xmfaFilebase.*>;
my $number_xmfaFile = $#xmfaFiles + 1;

for (my $blockid = 1; $blockid <= $number_xmfaFile; $blockid++)
{
  my $xmlFile = "$xmlFilebase.$blockid";
  my @e = split /\//, $xmlFile;
  my $xmlFilebasename = $e[$#e];
  if (-e $xmlFile)
  {
    my $parser = new XML::Parser();
    $parser->setHandlers(Start => \&startElement,
                         End => \&endElement,
                         Char => \&characterData,
                         Default => \&default);
    $itercount=0;
    my $doc;
    
    eval{ $doc = $parser->parsefile($xmlFile)};
    if ($@)
    {
      unlink $xmlFile; 
      print STDERR "Error in $blockid block $xmlFilebasename:$itercount\n";
      print "$xmlFilebasename:NP\n";
    }
    else
    {
      if ($params{samplesize} == $itercount)
      {
        print STDERR "Checked $blockid block $xmlFilebasename:$itercount\r";
      }
      else
      {
        unlink $xmlFile; 
        print STDERR "Error in $blockid block $xmlFilebasename:$itercount\n";
        print "$xmlFilebasename:NP\n";
      }
    }
  }
  else
  {
    print STDERR "Error in $blockid block $xmlFilebasename:$itercount\n";
    print "$xmlFilebasename:NE\n";
  }
}

exit;
##############################################################
# END OF RUN OF THIS PERL SCRIPT
##############################################################

#############################################################
# XML Parsing functions
#############################################################

sub startElement {
  my ($parseinst, $e, %attrs) = @_;
  $content = "";
  if ($e eq "Iteration") {
    $itercount++;
  }
  if ($e eq "outputFile") {
    # The start of XML file.
  }
}

sub endElement {
  my ($p, $elt) = @_;
  my $eltname;

  $content = "";
}

sub characterData {
  my( $parseinst, $data ) = @_;
  $data =~ s/\n|\t//g;

  $content .= $data;
}

sub default {
}

