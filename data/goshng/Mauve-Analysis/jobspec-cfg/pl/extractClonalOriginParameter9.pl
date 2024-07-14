#!/opt/local/bin/perl -w
#===============================================================================
#   Author: Sang Chul Choi, BSCB @ Cornell University, NY
#
#   File: extractClonalOriginParameter9.pl
#   Date: Thu Apr 21 12:42:10 EDT 2011
#   Version: 1.0
#
#   Usage:
#      perl extractClonalOriginParameter9.pl [options]
#
#      Try 'perl extractClonalOriginParameter9.pl -h' for more information.
#
#   Purpose: We parse a clonal origin XML file. 
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

my $VERSION = 'extractClonalOriginParameter9.pl 1.0';

my $man = 0;
my $help = 0;
my %params = ('help' => \$help, 'h' => \$help, 'man' => \$man);        
GetOptions( \%params,
            'help|h',
            'man',
            'verbose',
            'version' => sub { print $VERSION."\n"; exit; },
            'xml=s'
            ) or pod2usage(2);
pod2usage(1) if $help;
pod2usage(-exitstatus => 0, -verbose => 2) if $man;

=head1 NAME

extractClonalOriginParameter9.pl - Build a heat map of recombination.

=head1 VERSION

extractClonalOriginParameter9.pl 0.1.0

=head1 SYNOPSIS

perl extractClonalOriginParameter9.pl [-h] [-help] [-version] 
  [-xml xmlfile] 

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

=item B<-xml> <xmlfile>

A clonal origin XML file.

=back

=head1 AUTHOR

Sang Chul Choi, C<< <goshng_at_yahoo_dot_co_dot_kr> >>

=head1 BUGS

If you find a bug please post a message rnaseq_analysis project at codaset dot
com repository so that I can make extractClonalOriginParameter9.pl better.

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

my $xmlFile;

if (exists $params{xml})
{
  $xmlFile = $params{xml};
}
else
{
  &printError("you did not specify an XML file that contains Clonal Origin 2nd run results");
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
my @xmlFiles;
my @blockLength;
my @blocks;
my $tag;
my $content;
my %recedge;
my $itercount=0;

my $parser = new XML::Parser();
$parser->setHandlers(Start => \&startElement,
                     End => \&endElement,
                     Char => \&characterData,
                     Default => \&default);

$itercount=0;
my $doc;
eval{ $doc = $parser->parsefile($xmlFile)};
print "Unable to parse XML of $xmlFile, error $@\n" if $@;
 
exit;
##############################################################
# END OF RUN OF THIS PERL SCRIPT
##############################################################

#############################################################
# XML Parsing functions
#############################################################

sub startElement {
  my ($parseinst, $e, %attrs) = @_;
#  $tag = $element;
  $content = "";
  if ($e eq "Iteration") {
    $itercount++;

    for (my $i = 1; $i <= $#blocks; $i++)
    {
      my $xmlFileBlock = "$xmlFile.$i";
      my $f;
      open $f, ">>$xmlFileBlock" 
        or die "Could not open $xmlFileBlock";
      print $f "<Iteration>\n";
      close $f;
    }
  }
  if ($e eq "outputFile") {
    # The start of XML file.
  }
}

sub endElement {
  my ($p, $elt) = @_;
  my $eltname;
  if ($elt eq "Blocks") {
    @blocks = split /,/, $content;
    $numberBlock = $#blocks;
    for (my $i = 1; $i <= $#blocks; $i++)
    {
      my $xmlFileBlock = "$xmlFile.$i";
      my $f;
      open $f, ">$xmlFileBlock" 
        or die "Could not open $xmlFileBlock";
      # push @xmlFiles, $f;
      print $f "<?xml version = '1.0' encoding = 'UTF-8'?>\n";
      print $f "<outputFile>\n<Blocks>\n0,";
      my $d = $blocks[$i] - $blocks[$i-1];
      push @blockLength, $d;
      print $f $blockLength[$i-1];
      print $f "\n<\/Blocks>\n";
      close $f;
    }
    $Lt = $blocks[$#blocks];
  }

  if ($elt eq "comment") {
    for (my $i = 1; $i <= $#blocks; $i++)
    {
      my $xmlFileBlock = "$xmlFile.$i";
      my $f;
      open $f, ">>$xmlFileBlock" 
        or die "Could not open $xmlFileBlock";
      print $f "<comment>$content<\/comment>\n";
      close $f;
    }
  }

  if ($elt eq "nameMap") {
    for (my $i = 1; $i <= $#blocks; $i++)
    {
      my $xmlFileBlock = "$xmlFile.$i";
      my $f;
      open $f, ">>$xmlFileBlock" 
        or die "Could not open $xmlFileBlock";
      print $f "<nameMap>$content<\/nameMap>\n";
      close $f;
    }
  }

  $eltname = "regions";
  if ($elt eq $eltname) {
    my @regions = split /,/, $content;
    my $i = 0;

    for (my $i = 1; $i <= $#blocks; $i++)
    {
      my $xmlFileBlock = "$xmlFile.$i";
      my $f;
      open $f, ">>$xmlFileBlock" 
        or die "Could not open $xmlFileBlock";
      print $f "<$eltname>$regions[$i]<\/$eltname>\n";
      $i++;
      close $f;
    }
  }

################################# 8
    #for (my $i = 1; $i <= $#blocks; $i++)
    #{
      #my $xmlFileBlock = "$xmlFile.$i";
      #my $f;
      #open $f, ">>$xmlFileBlock" 
        #or die "Could not open $xmlFileBlock";
      #close $f;
    #}


  $eltname = "Tree";
  if ($elt eq $eltname) {
    for (my $i = 1; $i <= $#blocks; $i++)
    {
      my $xmlFileBlock = "$xmlFile.$i";
      my $f;
      open $f, ">>$xmlFileBlock" 
        or die "Could not open $xmlFileBlock";
      print $f "<$eltname>\n$content\n<\/$eltname>\n";
      close $f;
    }
  }

  print_xml_one_line ("number", $elt);
  print_xml_one_line ("ll", $elt);
  print_xml_one_line ("prior", $elt);

  $eltname = "theta";
  if ($elt eq $eltname) {
    for (my $i = 1; $i <= $#blocks; $i++)
    {
      my $xmlFileBlock = "$xmlFile.$i";
      my $f;
      open $f, ">>$xmlFileBlock" 
        or die "Could not open $xmlFileBlock";
      my $Lb = $blockLength[$i-1];
      my $thetaPerBlock = $content / $Lt * $Lb;
      print $f "<$eltname>$thetaPerBlock<\/$eltname>\n";
      close $f;
    }
  }

  $eltname = "rho";
  if ($elt eq $eltname) {
    $rho = $content;
  }

  $eltname = "delta";
  if ($elt eq $eltname) {
    $delta = $content;

    # Print rho's and delta.
    for (my $i = 1; $i <= $#blocks; $i++)
    {
      my $xmlFileBlock = "$xmlFile.$i";
      my $f;
      open $f, ">>$xmlFileBlock" 
        or die "Could not open $xmlFileBlock";
      my $Lb = $blockLength[$i-1];
      my $rhoPerBlock = $rho / ($Lt + $numberBlock*($delta - 1)) * ($Lb + $delta - 1);
      print $f "<rho>$rhoPerBlock<\/rho>\n";
      print $f "<$eltname>$delta<\/$eltname>\n";
      close $f;
    }
  }

  print_xml_one_line ("tmrca", $elt);
  print_xml_one_line ("esttheta", $elt);
  print_xml_one_line ("estvartheta", $elt);
  print_xml_one_line ("estrho", $elt);
  print_xml_one_line ("estvarrho", $elt);
  print_xml_one_line ("estdelta", $elt);
  print_xml_one_line ("estvardelta", $elt);
  print_xml_one_line ("estnumrecedge", $elt);
  print_xml_one_line ("estvarnumrecedge", $elt);
  print_xml_one_line ("estedgeden", $elt);
  print_xml_one_line ("estvaredgeden", $elt);
  print_xml_one_line ("estedgepb", $elt);
  print_xml_one_line ("estvaredgepb", $elt);
  print_xml_one_line ("estedgevarpb", $elt);
  print_xml_one_line ("estvaredgevarpb", $elt);

  $eltname = "recedge";
  if ($elt eq $eltname) {
    my $startBlock;
    my $endBlock;
    # 0   10000    20000   30000
    # 0..9999 
    # 10000..19999
    # 20000..29999
    for (my $i = 0; $i <= $#blocks; $i++)
    {
      if ($recedge{start} < $blocks[$i])
      {
        $startBlock = $i;
        last;
      }
    }
    for (my $i = 0; $i <= $#blocks; $i++)
    {
      if ($recedge{end} <= $blocks[$i])
      {
        $endBlock = $i;
        last;
      }
    }
    if ($startBlock == $endBlock) {
      my $start = $recedge{start} - $blocks[$startBlock-1];
      my $end = $recedge{end} - $blocks[$startBlock-1];

      # my $f = $xmlFiles[$startBlock-1];

      my $xmlFileBlock = "$xmlFile.$startBlock";
      my $f;
      open $f, ">>$xmlFileBlock" 
        or die "Could not open $xmlFileBlock";
      print $f "<recedge>";
      print $f "<start>$start<\/start>";
      print $f "<end>$end<\/end>";
      print $f "<efrom>$recedge{efrom}<\/efrom>";
      print $f "<eto>$recedge{eto}<\/eto>";
      print $f "<afrom>$recedge{afrom}<\/afrom>";
      print $f "<ato>$recedge{ato}<\/ato>";
      print $f "<\/recedge>\n";
      close $f;
    } else {
      die "$startBlock == $endBlock must be the same";
      my $start = $recedge{start} - $blocks[$startBlock-1];
      my $end = $blocks[$startBlock] - $blocks[$startBlock-1];

      # my $f = $xmlFiles[$startBlock-1];

      my $xmlFileBlock = "$xmlFile.$startBlock";
      my $f;
      open $f, ">>$xmlFileBlock" 
        or die "Could not open $xmlFileBlock";
      print $f "<recedge>";
      print $f "<start>$start<\/start>";
      print $f "<end>$end<\/end>";
      print $f "<efrom>$recedge{efrom}<\/efrom>";
      print $f "<eto>$recedge{eto}<\/eto>";
      print $f "<afrom>$recedge{afrom}<\/afrom>";
      print $f "<ato>$recedge{ato}<\/ato>";
      print $f "<\/recedge>\n";
      close $f;

      for (my $i = 0; $i < $endBlock - $startBlock - 1; $i++)
      {
        my $start = 0;
        my $end = $blocks[$startBlock + 1 + $i] - $blocks[$startBlock + $i];

        # my $f = $xmlFiles[$startBlock-1];
        my $xmlFileBlock = "$xmlFile.$startBlock";
        my $f;
        open $f, ">>$xmlFileBlock" 
          or die "Could not open $xmlFileBlock";
        print $f "<recedge>";
        print $f "<start>$start<\/start>";
        print $f "<end>$end<\/end>";
        print $f "<efrom>$recedge{efrom}<\/efrom>";
        print $f "<eto>$recedge{eto}<\/eto>";
        print $f "<afrom>$recedge{afrom}<\/afrom>";
        print $f "<ato>$recedge{ato}<\/ato>";
        print $f "<\/recedge>\n";
        close $f;
      }

      $start = $blocks[$endBlock] - $blocks[$endBlock-1];
      $end = $recedge{end} - $blocks[$endBlock-1];
      #$f = $xmlFiles[$endBlock-1];
      $xmlFileBlock = "$xmlFile.$endBlock";
      #my $f;
      open $f, ">>$xmlFileBlock" 
        or die "Could not open $xmlFileBlock";
      print $f "<recedge>";
      print $f "<start>$start<\/start>";
      print $f "<end>$end<\/end>";
      print $f "<efrom>$recedge{efrom}<\/efrom>";
      print $f "<eto>$recedge{eto}<\/eto>";
      print $f "<afrom>$recedge{afrom}<\/afrom>";
      print $f "<ato>$recedge{ato}<\/ato>";
      print $f "<\/recedge>\n";
      close $f;
    }
  }

  $eltname = "Iteration";
  if ($elt eq $eltname) {
    for (my $i = 1; $i <= $#blocks; $i++)
    {
      my $xmlFileBlock = "$xmlFile.$i";
      my $f;
      open $f, ">>$xmlFileBlock" 
        or die "Could not open $xmlFileBlock";
      print $f "<\/$eltname>\n";
      close $f;
    }
  }

  $eltname = "outputFile";
  if ($elt eq $eltname) {
    for (my $i = 1; $i <= $#blocks; $i++)
    {
      my $xmlFileBlock = "$xmlFile.$i";
      my $f;
      open $f, ">>$xmlFileBlock" 
        or die "Could not open $xmlFileBlock";
      print $f "<\/$eltname>\n";
      close $f;
    }
  }

  if ($elt eq "start") {
    $recedge{start} = $content;
  }
  if ($elt eq "end") {
    $recedge{end} = $content;
  }
  if ($elt eq "efrom") {
    $recedge{efrom} = $content;
  }
  if ($elt eq "eto") {
    $recedge{eto} = $content;
  }
  if ($elt eq "afrom") {
    $recedge{afrom} = $content;
  }
  if ($elt eq "ato") {
    $recedge{ato} = $content;
  }

  if ($elt eq "recedge")
  {
  }
  
#  $tag = "";
  $content = "";
}

sub characterData {
  my( $parseinst, $data ) = @_;
  $data =~ s/\n|\t//g;

  $content .= $data;
}

sub default {
}

##
#################################################################################
### MISC FUNCTIONS
#################################################################################
##

sub printError {
    my $msg = shift;
    print STDERR "ERROR: ".$msg.".\n\nTry \'extractClonalOriginParameter9.pl -h\' for more information.\nExit program.\n";
    exit(0);
}

sub getLineNumber {
    my $file = shift;
    my $lines = 0;
    open(FILE,"perl -p -e 's/\r/\n/g' < $file |") or die "ERROR: Could not open file $file: $! \n";
    $lines += tr/\n/\n/ while sysread(FILE, $_, 2 ** 16);
    close(FILE);
    return $lines;
}

sub print_xml_one_line ($$) {
  my ($eltname, $elt) = @_;
  if ($elt eq $eltname) {

    for (my $i = 1; $i <= $#blocks; $i++)
    {
      my $xmlFileBlock = "$xmlFile.$i";
      my $f;
      open $f, ">>$xmlFileBlock" 
        or die "Could not open $xmlFileBlock";
      print $f "<$eltname>$content<\/$eltname>\n";
      close $f;
    }
  }
}

