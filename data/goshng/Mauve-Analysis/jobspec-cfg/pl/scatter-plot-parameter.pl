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
use File::Temp qw(tempfile);
require "pl/sub-xmfa.pl";

$| = 1; # Do not buffer output
my $VERSION = 'scatter-plot-parameter.pl 1.0';
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
            'version' => sub { print $VERSION."\n"; exit; },
            'xmlbase=s',
            'xmfabase=s',
            'in=s',
            'out=s',
            '<>' => \&process
            ) or pod2usage(2);
pod2usage(1) if $help;
pod2usage(-exitstatus => 0, -verbose => 2) if $man;

my $xmlFilebase;
my $xmfaFilebase;
my $basenameOutFile;

if (exists $params{xmlbase})
{
  $xmlFilebase = $params{xmlbase};
}

if (exists $params{xmfabase})
{
  $xmfaFilebase = $params{xmfabase};
}

if (exists $params{out})
{
  $basenameOutFile = $params{out};
}
else
{
  &printError("you did not specify a base name of the output file");
}

if ($cmd eq "three")
{
  unless (exists $params{xmfabase} and exists $params{xmlbase})
  {
    &printError("Command $cmd requires options -xmfabase and -xmlbase");
  }
}
elsif ($cmd eq "bed")
{
  unless (exists $params{in})
  {
    &printError("Command $cmd requires option -in");
  }
}

################################################################################
## DATA PROCESSING
################################################################################
##############################################################
# Global variables
##############################################################
my $tag;
my $content;
my $itercount=0;
my $rhoPerBlock; 
my $position;
my $blockID;
my $blockLength;

if ($cmd eq "three")
{
  ##############################################################
  # Open the three output files.
  ##############################################################
  open OUTTHETA, ">$basenameOutFile-theta" or die $!;
  open OUTRHO, ">$basenameOutFile-rho" or die $!;
  open OUTDELTA, ">$basenameOutFile-delta" or die $!;

  ##############################################################
  # Start to parse the XML file.
  ##############################################################

  my @xmlFiles = <$xmlFilebase.xml.*>;
  foreach my $xmlFile (@xmlFiles) {

    $xmlFile =~ /\.xml\.(\d+)/;
    $blockID = $1;
    $position = getMidPosition ("$xmfaFilebase.$blockID");
    my $parser = new XML::Parser();
    $parser->setHandlers(Start => \&startElement,
                         End => \&endElement,
                         Char => \&characterData,
                         Default => \&default);

    my $doc;
    $itercount = 0;
    eval{ $doc = $parser->parsefile($xmlFile)};
    print "Unable to parse XML of $xmlFile, error $@\n" if $@;
  }

  close OUTTHETA;
  close OUTRHO;
  close OUTDELTA;
}
elsif ($cmd eq "bed")
{
  open OUT, ">", $params{out} or die "cannot open > $params{out} $!";
  my $prevPos;
  my $pos = -1;
  my $value = 0;
  my $count = 0;
  my $isFirst = 1;
  open IN, $params{in} or die "cannot open < $params{in} $!";
  while (<IN>)
  {
    chomp;
    my @e = split /\s+/;
    $pos = $e[0];
    if ($isFirst == 1)
    {
      $isFirst = 0;
      $count++;
      $value = $e[1];
    }
    elsif ($pos == $prevPos)
    {
      $count++;
      $value += $e[1];
    }
    else
    {
      my $name = int($pos);
      my $start = $name - 3000;
      if ($start < 0)
      {
        $start = 0;
      }
      my $end = $name;
      # $value /= $count;
      $value *= 200;
      if ($pos < 1900000)
      {
      print OUT "chr1\t$start\t$end\t$name\t$value\n";
      #print OUT "chr1\t$start\t$end\n";
      }
      $count = 0;
      $value = $e[1];
    }
    $prevPos = $pos;
  }
  close IN;
  close OUT;
}
exit;
##############################################################
# END OF RUN OF THIS PERL SCRIPT
##############################################################

##############################################################
# XML Processing procedures
##############################################################
sub startElement {
  my( $parseinst, $element, %attrs ) = @_;
  $tag = $element;
  $content = "";
  SWITCH: {
    if ($element eq "Iteration") {
      $itercount++;
      last SWITCH;
    }
    if ($element eq "theta") {
      last SWITCH;
    }
    if ($element eq "delta") {
      last SWITCH;
    }
    if ($element eq "rho") {
      last SWITCH;
    }
  }
}

sub endElement {
  my ($p, $elt) = @_;
  if($tag eq "theta"){
    my $thetaPerSite = $content / $blockLength;
    # print OUTTHETA "$blockID\t$thetaPerSite\n";
    print OUTTHETA "$position\t$thetaPerSite\t$blockID\n";
  }
  if($tag eq "rho"){
    $rhoPerBlock = $content;
  }
  if($tag eq "delta"){
    print OUTDELTA "$position\t$content\t$blockID\n";
    my $rhoPerSite = $rhoPerBlock / ($blockLength + $content - 1);
    print OUTRHO "$position\t$rhoPerSite\t$blockID\n";
  }
  $tag = "";
  $content = "";
}

sub characterData {
  my( $parseinst, $data ) = @_;
  $data =~ s/\n|\t//g;
  if($tag eq "theta"){
    $content .= $data;
  }
  if($tag eq "rho"){
    $content .= $data;
  }
  if($tag eq "delta"){
    $content .= $data;
  }
  if ($tag eq "Blocks") {
    if (length($data) > 1) {
      # print "$tag:$data\n";
      $data =~ s/.+\,//g;
      $blockLength = $data;
      # print "$blockLength\n";
      die "block length is not a positive integer ($blockLength)" unless $blockLength > 0;
    }
  }
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
  print STDERR "ERROR: ".$msg.".\n\nTry \'scatter-plot-parameter.pl -h\' for more information.\nExit program.\n";
  exit(0);
}
__END__
=head1 NAME

scatter-plot-parameter.pl - File for plotting three population parameter estimates

=head1 VERSION

scatter-plot-parameter.pl 1.0

=head1 SYNOPSIS

perl pl/scatter-plot-parameter.pl three -xmlbase path/to/co1output -xmfabase filepath/to/corexmfa

perl pl/scatter-plot-parameter.pl bed -in scatter-plot-parameter-1-out-rho

=head1 DESCRIPTION

The three scalar parameters include mutation rate, recombination rate, and
average recombinant tract length.  These values are extracted from 
ClonalOrigin XML output files.

=head1 OPTIONS

=over 8

=item B<-help> | B<-h>

Print the help message; ignore other arguments.

=item B<-man>

Print the full documentation; ignore other arguments.

=item B<-version>

Print program version; ignore other arguments.

=item B<***** INPUT OPTIONS *****>

=item B<-xmlbase> <file>

A prefix of ClonalOrigin output files in XML format is required.

=item B<-out> <base name of output file>

Three files are generated for the three parameters. The option string of out is
used as a base name of them: i.e., out-theta, out-rho, and out-delta.

=back

=head1 AUTHOR

Sang Chul Choi, C<< <goshng_at_yahoo_dot_co_dot_kr> >>

=head1 BUGS

If you find a bug please post a message rnaseq_analysis project at codaset dot
com repository so that I can make scatter-plot-parameter.pl better.

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
