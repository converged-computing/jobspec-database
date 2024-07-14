#!/usr/bin/perl
#===============================================================================
#   Author: Sang Chul Choi, BSCB @ Cornell University, NY
#
#   File: convert-gff-ingene.pl
#   Date: Fri May  6 21:59:36 EDT 2011
#   Version: 1.0
#===============================================================================

use strict;
use warnings;

use Getopt::Long;
use Pod::Usage;

$| = 1; # Do not buffer output

my $VERSION = 'convert-gff-ingene.pl 1.0';

my $man = 0;
my $help = 0;
my %params = ('help' => \$help, 'h' => \$help);        
GetOptions( \%params,
            'help|h',
            'verbose',
            'version' => sub { print $VERSION."\n"; exit; },
            'gff=s',
            'withdescription',
            'out=s'
            ) or pod2usage(2);
pod2usage(1) if $help;
pod2usage(-exitstatus => 0, -verbose => 2) if $man;

=head1 NAME

convert-gff-ingene.pl - Parse a gff file to make ingene file

=head1 VERSION

convert-gff-ingene.pl 1.0

=head1 SYNOPSIS

perl convert-gff-ingene.pl.pl [-h] [-help] [-version] [-verbose]
  [-gff file] 
  [-out file] 

=head1 DESCRIPTION

convert-gff-ingene.pl parses a gff file. A gff file is a companion file of a
genbank bacterial genome file. An ingene file is a simpler file where each row
shows gene name, gene start position, gene end position, and gene strand
delimted by tab.

=head1 OPTIONS

=over 8

=item B<-help> | B<-h>

Print the help message; ignore other arguments.

=item B<-version>

Print program version; ignore other arguments.

=item B<-verbose>

Prints status and info messages during processing.

=item B<***** INPUT OPTIONS *****>

=item B<-gff> <file>

A gff file name.

=item B<-out> <file>

An output file.

=back

=head1 AUTHOR

Sang Chul Choi, C<< <goshng_at_yahoo_dot_co_dot_kr> >>

=head1 BUGS

If you find a bug please post a message rnaseq_analysis project at codaset dot
com repository so that I can make convert-gff-ingene.pl better.

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

################################################################################
## COMMANDLINE OPTION PROCESSING
################################################################################

my $gff;
my $out;

if (exists $params{gff}) {
  $gff = $params{gff};
} else {
  &printError("you did not specify a gff file name");
}

if (exists $params{out}) {
  $out = $params{out};
} else {
  &printError("you did not specify an output file name");
}

################################################################################
## DATA PROCESSING
################################################################################

sub convert_gff_ingene($$); 

convert_gff_ingene($gff, $out); 

sub convert_gff_ingene($$) {
  my ($gffFilename, $out) = @_;
  open OUT, ">$out" or die "$out could not be opened"; 

  my $numGene = 0;
  my $gffLine;
  my $blockid_prev = 0;
  open GFF, "$gffFilename" or die $!;
  while ($gffLine = <GFF>)
  {
    if (exists $params{withdescription})
    {
      if ($gffLine =~ /RefSeq\s+CDS\s+(\d+)\s+(\d+).+([+-]).+locus_tag=(\w+);.+;product=(.+);/)
      {
        my @e = split (/;/, $5);
        print OUT "$4\t$e[0]\n";
      }
    }
    else
    {
      if ($gffLine =~ /RefSeq\s+gene\s+(\d+)\s+(\d+).+([+-]).+locus_tag=(\w+);/)
      {
        print OUT "$4\t$1\t$2\t$3\n";
      }
    }
  }
  close GFF;
  close OUT;
}
