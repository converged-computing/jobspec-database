use strict;
use warnings;

my $map = "p2P.s";
$map = $ARGV[1] if defined $ARGV[1];
my %p2P;
my %bad;
open A, "zcat $map|";
while (<A>){
  chop();
  my ($f, $t, $b, $b1) = split(/;/);
  if ($b > 0 || $b1 > 0){
    $bad{$f}++;
  }else{
    $p2P{$f} = $t if $f ne $t; # don't store identity
  }
}

my @cols = ($ARGV[0]+0);
for my $i (2..$#ARGV){
  push @cols, $ARGV[$i];
}
  
while(<STDIN>){
  chop();
  my @x=split(/;/);
  my $pr = 0;
  for my $col (@cols){
    if (defined $x[$col] && !defined $bad{$x[$col]}){
      $pr ++;
      $x[$col] = $p2P{$x[$col]} if defined $p2P{$x[$col]};
    }
  }
  print "".(join ';', @x)."\n" if $pr == $#cols+1;
}
