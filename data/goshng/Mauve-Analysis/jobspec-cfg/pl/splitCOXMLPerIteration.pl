#!/opt/local/bin/perl -w
#===============================================================================
#   Author: Sang Chul Choi, BSCB @ Cornell University, NY
#
#   File: splitCOXMLPerIteration.pl
#   Date: Thu Apr 21 12:42:10 EDT 2011
#   Version: 1.0
#
#   Usage:
#      perl splitCOXMLPerIteration.pl [options]
#
#      Try 'perl splitCOXMLPerIteration.pl -h' for more information.
#
#   Purpose: For each block I grab an <Iteration>. Pull all of the Iteration to
#            make an XML file with a single clonal frame with its recombinant
#            edges. 
#            1. Find block lengths.
#
#   Note that I started to code this based on PRINSEQ by Robert SCHMIEDER at
#   Computational Science Research Center @ SDSU, CA as a template. Some of
#   words are his not mine, and credit should be given to him. 
#===============================================================================

use strict;
use warnings;
use XML::Parser;
use Getopt::Long;
use Pod::Usage;
use File::Temp qw(tempfile);

$| = 1; # Do not buffer output

my $VERSION = 'splitCOXMLPerIteration.pl 1.0';

my $man = 0;
my $help = 0;
my %params = ('help' => \$help, 'h' => \$help, 'man' => \$man);        
GetOptions( \%params,
            'help|h',
            'man',
            'verbose',
            'check',
            'version' => sub { print $VERSION."\n"; exit; },
            'd=s',
            'outdir=s',
            'xmlbasename=s',
            'numberblock=i',
            'endblockid'
            ) or pod2usage(2);
pod2usage(1) if $help;
pod2usage(-exitstatus => 0, -verbose => 2) if $man;

=head1 NAME

splitCOXMLPerIteration.pl - Build a heat map of recombination.

=head1 VERSION

splitCOXMLPerIteration.pl 0.1.0

=head1 SYNOPSIS

perl splitCOXMLPerIteration.pl [-h] [-help] [-version] 
  [-xml xmlfile] 
  [-xmlbasename filename]

=head1 DESCRIPTION

An XML Clonal Origin file is divided into files for multiple blocks.

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

=item B<-d> <clonal origin output2>

A directory that contains clonal origin XML output files of the 2nd stage.

=item B<-n> <number of subsample>

An integer number. Default is 10.

=back

=head1 AUTHOR

Sang Chul Choi, C<< <goshng_at_yahoo_dot_co_dot_kr> >>

=head1 BUGS

If you find a bug please post a message rnaseq_analysis project at codaset dot
com repository so that I can make splitCOXMLPerIteration.pl better.

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

my $xmlDir;
my $xmlBasename = "core_co.phase3";
my $endblockid = 0;
my $check = 0;
my $outDir;
my $numberBlock;
my $verbose = 0;

if (exists $params{d})
{
  $xmlDir= $params{d};
}
else
{
  &printError("you did not specify an XML file directory that contains Clonal Origin 2nd run results");
}

if (exists $params{outdir})
{
  $outDir = $params{outdir};
}
else
{
  &printError("you did not specify the output directory");
}

if (exists $params{numberblock})
{
  $numberBlock = $params{numberblock};
}
else
{
  &printError("you did not specify the number of blocks");
}

if (exists $params{endblockid})
{
  $endblockid = 1;
}

if (exists $params{verbose})
{
  $verbose = 1;
}

if (exists $params{check})
{
  $check = 1;
}

if (exists $params{xmlbasename})
{
  $xmlBasename = $params{xmlbasename};
}

#
################################################################################
## DATA PROCESSING
################################################################################
#

##############################################################
# Global variables
##############################################################
my $iterationContent;
my $sampleIter;
my $offset;
my $treeFile;

my $delta;
my $rho;
my $Lt;           # Total length of all of the blocks
my @blocks;
my $tag;
my $content;
my %recedge;
my $itercount=0;


##############################################################
# Find number of blocks.
##############################################################
my @xmlFiles;
if ($endblockid == 1)
{
  @xmlFiles =  <$xmlDir/$xmlBasename.xml.*>;
}
else
{
  @xmlFiles =  <$xmlDir/$xmlBasename.*.xml>;
}
my $numBlocks = $#xmlFiles + 1;

##############################################################
# Find the sample size of an Clonal Origin XML.
##############################################################
my $xmlfilename = "$xmlDir/$xmlBasename.1.xml";
if ($endblockid == 1)
{
  $xmlfilename = "$xmlDir/$xmlBasename.xml.1";
}
my $sampleSizeFirst = get_sample_size ($xmlfilename);
for (my $blockid = 2; $blockid <= $numBlocks; $blockid++)
{
  my $xmlfilename = "$xmlDir/$xmlBasename.$blockid.xml";
  if ($endblockid == 1)
  {
    $xmlfilename = "$xmlDir/$xmlBasename.xml.$blockid";
  }

  my $sampleSize = get_sample_size ($xmlfilename);
  die "The first block ($sampleSizeFirst) and the $blockid-th block ($sampleSize) are different"
    unless $sampleSizeFirst == $sampleSize;
}

if ($check == 1)
{
  print "sampleSizeFirst:$sampleSizeFirst\n";
}

##############################################################
# Find the total length of all the blocks.
##############################################################
my $totalLength = 0;
my @blockLengths;
push @blockLengths, 0;
for (my $blockid = 1; $blockid <= $numBlocks; $blockid++)
{
  my $xmlfilename = "$xmlDir/$xmlBasename.$blockid.xml";
  if ($endblockid == 1)
  {
    $xmlfilename = "$xmlDir/$xmlBasename.xml.$blockid";
  }

  my $blockLength = get_block_length ($xmlfilename);
  $totalLength += $blockLength;
  push @blockLengths, $totalLength;
}
if ($check == 1)
{
  print "totalLength:$totalLength\n";
}

#for (my $blockid = 1; $blockid <= $numBlocks; $blockid++)
#{
  #my $xmlFile = "$xmlDir/$xmlBasename.xml.$blockid";
  #my $header = get_header_coxml ($xmlFile);
  #for (my $iterationid = 1; $iterationid <= $sampleSizeFirst; $iterationid++)
  #{
    #my $outXmlFile = "$outDir/$xmlBasename.xml.$blockid.$iterationid";
    #open OUTXML, ">$outXmlFile" or die "$xmlFile could not be opended";
    #print OUTXML $header;
    #close OUTXML;
  #}
  #if ($verbose == 1)
  #{
    #print STDERR "Block: $blockid\r";
  #}
#}

for (my $blockid = 1; $blockid <= $numBlocks; $blockid++)
{
  my $xmlFile = "$xmlDir/$xmlBasename.xml.$blockid";
  my $line;
  my $header = ""; 
  open XML, $xmlFile or die "$xmlFile could not be opened";
  
  my $iterationid = 0;
  while ($line = <XML>)
  {
    if ($line =~ /^<Iteration>/)
    {
      if ($iterationid > 0)
      {
        print OUTXML "</outputFile>\n";
        close OUTXML;
        if ($verbose == 1)
        {
          print STDERR "Block: $blockid - $iterationid\r";
        }
      }
      $iterationid++; 
      my $outXmlFile = "$outDir/$xmlBasename.xml.$blockid.$iterationid";
      open OUTXML, ">$outXmlFile" or die "$xmlFile could not be opended";
      print OUTXML $header;
    }
    if ($iterationid > 0)
    {
      print OUTXML $line;
    }
    else
    {
      $header .= $line;
    }
  }
  
  if ($verbose == 1)
  {
    print STDERR "Block: $blockid\r";
  }
  close XML;
}

exit;

