#!/opt/local/bin/perl -w
use strict;
#use IO::Uncompress::Bunzip2 qw(bunzip2 $Bunzip2Error);
use XML::Parser;

if(@ARGV==0){
	die "Usage: computeMedians.pl <ClonalOrigins XML or xml.bz2>\n";
}

my @lens;
my @meantheta;
my @meandelta;
my @meanrho;
my $itercount=0;
my $curtheta=0;
my $curdelta=0;
my $currho=0;
my $tag;

my $blockcount=scalar(@ARGV);	# assume one block per file

# extract posterior mean estimates of global parameters from each file
foreach my $f (@ARGV){
	my $fs;
	if($f =~ /\.bz2$/){
		#$fs = bunzip2 $f => "tmpxml" or die "IO::Uncompress::Bunzip2 failed: $Bunzip2Error\n";
		#$fs = "tmpxml";
	}else{
		$fs = $f;
	}
	my $parser = new XML::Parser();

	$parser->setHandlers(      Start => \&startElement,
                           End => \&endElement,
                           Char => \&characterData,
                           Default => \&default);

	$itercount=0;
	$curtheta=0;
	$curdelta=0;
	$currho=0;
	my $doc;
	eval{ $doc = $parser->parsefile($fs)};
	print "Unable to parse XML of $f, error $@\n" if $@;
	next if $@;
	print "parsed $f\n";
	$curtheta /= $itercount;
	$curdelta /= $itercount;
	$currho /= $itercount;
	push( @meantheta, $curtheta );
	push( @meandelta, $curdelta );
	push( @meanrho, $currho );
}

# convert to per-site values of theta and rho
for( my $i=0; $i<@meantheta; $i++){
	$meantheta[$i] /= $lens[$i];
	$meanrho[$i] /= $meandelta[$i] + $lens[$i];
}

# now compute a weighted median
my %thetalens;
my %deltalens;
my %rholens;
my $lensum=0;
for( my $i=0; $i<@meantheta; $i++){
	$thetalens{$meantheta[$i]}=$lens[$i];
	$deltalens{$meandelta[$i]}=$lens[$i];
	$rholens{$meanrho[$i]}=$lens[$i];
	$lensum += $lens[$i];
}
print "lensum is $lensum\n";


my @tsort = sort{ $a <=> $b } @meantheta;
my @dsort = sort{ $a <=> $b } @meandelta;
my @rsort = sort{ $a <=> $b } @meanrho;

# Find the weighted median of theta
my $j=0;
for(my $ttally=$thetalens{$tsort[$j]}; $ttally < $lensum/2; $ttally += $thetalens{$tsort[$j]})
{
  $j++;
}
my $weightMedianTheta = $tsort[$j];

# Find the weighted Q1 of theta
$j=0;
for(my $ttally=$thetalens{$tsort[$j]}; $ttally < $lensum/4; $ttally += $thetalens{$tsort[$j]})
{
  $j++;
}
my $weightQ1Theta = $tsort[$j];

# Find the weighted Q3 of theta
$j=0;
for(my $ttally=$thetalens{$tsort[$j]}; $ttally < 3*$lensum/4; $ttally += $thetalens{$tsort[$j]})
{
  $j++;
}
my $weightQ3Theta = $tsort[$j];

print "Median theta:$weightMedianTheta\n";
print "Median Q1 theta:$weightQ1Theta\n";
print "Median Q3 theta:$weightQ3Theta\n";

# Find the weighted median of delta
$j=0;
for(my $dtally=$deltalens{$dsort[$j]}; $dtally < $lensum/2; $dtally += $deltalens{$dsort[$j]})
{
  $j++;
}
my $weightMedianDelta = $dsort[$j];

# Find the weighted Q1 of delta
$j=0;
for(my $dtally=$deltalens{$dsort[$j]}; $dtally < $lensum/4; $dtally += $deltalens{$dsort[$j]})
{
  $j++;
}
my $weightQ1Delta = $dsort[$j];

# Find the weighted Q3 of delta
$j=0;
for(my $dtally=$deltalens{$dsort[$j]}; $dtally < 3 * $lensum/4; $dtally += $deltalens{$dsort[$j]})
{
  $j++;
}
my $weightQ3Delta = $dsort[$j];

print "Median delta:$weightMedianDelta\n";
print "Median Q1 delta:$weightQ1Delta\n";
print "Median Q3 delta:$weightQ3Delta\n";

# Find the weighted median of delta
$j=0;
for(my $rtally=$rholens{$rsort[$j]}; $rtally < $lensum/2; $rtally += $rholens{$rsort[$j]})
{
  $j++;
}
my $weightMedianRho = $rsort[$j];

# Find the weighted Q1 of delta
$j=0;
for(my $rtally=$rholens{$rsort[$j]}; $rtally < $lensum/4; $rtally += $rholens{$rsort[$j]})
{
  $j++;
}
my $weightQ1Rho = $rsort[$j];

# Find the weighted median of delta
$j=0;
for(my $rtally=$rholens{$rsort[$j]}; $rtally < 3 * $lensum/4; $rtally += $rholens{$rsort[$j]})
{
  $j++;
}
my $weightQ3Rho = $rsort[$j];

print "Median rho:$weightMedianRho\n";
print "Median Q1 rho:$weightQ1Rho\n";
print "Median Q3 rho:$weightQ3Rho\n";

exit;

#################################################################################

sub startElement {
       my( $parseinst, $element, %attrs ) = @_;
	$tag = $element;
       SWITCH: {
              if ($element eq "Iteration") {
                     $itercount++;
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
	$tag = "";
}

sub characterData {
       my( $parseinst, $data ) = @_;
	$data =~ s/\n|\t//g;
	$curtheta += $data if ($tag eq "theta");
	$curdelta += $data if ($tag eq "delta");
	$currho += $data if ($tag eq "rho");
	if($tag eq "Blocks"){
		$data =~ s/.+\,//g;
		push( @lens, $data ) if(length($data)>1);
	}
}

sub default {
}
