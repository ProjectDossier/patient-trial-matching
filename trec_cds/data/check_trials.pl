#!/usr/bin/perl -w

use strict;

# Check a TREC 2021 Clinical Trials track submission for various
# common errors:
#      * extra fields
#      * multiple run tags
#      * missing or extraneous topics
#      * invalid retrieved documents (approximate check)
#      * duplicate retrieved documents in a single topic
#      * too many documents retrieved for a topic
# Messages regarding submission are printed to an error log
# Task uses a completely standard TREC input file

# Results input file is in the form
#     topic_num Q0 docid rank sim tag

# This script differs from the version released to TREC participants
# in that it creates a new file, "input", in the current directory
# that has a more standard format (topic numbers are guaranteed not to
# have leading 0's; the second field will be exactly "Q0"; there will
# not be leading whitespace and columns will be separated by one tab)

# Change these variable values to the directory in which the error log should be put
my $errlog_dir = ".";

# If more than MAX_ERRORS errors, then stop processing; something drastically
# wrong with the file.
my $MAX_ERRORS = 25; 
# May return up to MAX_RET visit ids per topic
my $MAX_RET = 1000;

my %valid_ids;
my @topics;
my %numret;                     # number of docs retrieved per topic
my $results_file;    		# input param: file to be checked
my $errlog;                     # file name of error log
my ($q0warn, $num_errors);      # flags for errors detected
my $line;                       # current input line
my ($topic,$q0,$docno,$rank,$sim,$tag);
my $line_num;                   # current input line number
my $run_id;
my ($i,$t,$last_i);

my $usage = "Usage: $0 resultsfile\n";
$#ARGV == 0 || die $usage;
$results_file = $ARGV[0];

@topics = 1 .. 75;
my $mintopic = $topics[0];
my $maxtopic = $topics[scalar(@topics)-1];

open RESULTS, "<$results_file" ||
    die "Unable to open results file $results_file: $!\n";

$last_i = -1;
while ( ($i=index($results_file,"/",$last_i+1)) > -1) {
    $last_i = $i;
}
$errlog = $errlog_dir . "/" . substr($results_file,$last_i+1) . ".errlog";
open ERRLOG, ">$errlog" ||
    die "Cannot open error log for writing\n";

for my $t (@topics) {
    $numret{$t} = 0;
}
$q0warn = 0;
$num_errors = 0;
$line_num = 0;
$run_id = "";

while ($line = <RESULTS>) {
    chomp $line;
    next if ($line =~ /^\s*$/);

    undef $tag;
    my @fields = split " ", $line;
    $line_num++;
	
    if (scalar(@fields) == 6) {
	($topic,$q0,$docno,$rank,$sim,$tag) = @fields;
    } else {
	&error("Wrong number of fields (expecting 6)");
	exit 255;
    }
	
    # make sure runtag is ok
    if (! $run_id) {		# first line --- remember tag 
	$run_id = $tag;
	if ($run_id !~ /^[A-Za-z0-9_]{1,12}$/) {
	    &error("Run tag `$run_id' is malformed (must be 1-12 alphanumeric characters)");
	    next;
	}
    }
    else {		       # otherwise just make sure one tag used
	if ($tag ne $run_id) {
	    &error("Run tag inconsistent (`$tag' and `$run_id')");
	    next;
	}
    }
	
    # get topic number
    if ($topic < $mintopic || $topic > $maxtopic) {
	&error("Unknown topic ($topic)");
	$topic = 0;
	next;
    }
	
	
    # make sure second field is "Q0"
    if ($q0 ne "Q0" && ! $q0warn) {
	$q0warn = 1;
	&error("Field 2 is `$q0' not `Q0'");
    }
    
    # remove leading 0's from rank (but keep final 0!)
    $rank =~ s/^0*//;
    if (! $rank) {
	$rank = "0";
    }
	
    # make sure rank is an integer (a past group put sim in rank field by accident)
    if ($rank !~ /^[0-9-]+$/) {
	&error("Column 4 (rank) `$rank' must be an integer");
    }
	
    # make sure DOCNO has right format and not duplicated
    if (check_docno($docno)) {
	if (exists $valid_ids{$docno} && $valid_ids{$docno} == $topic){
	    &error("$docno retrieved more than once for topic $topic");
	    next;
	}
	$valid_ids{$docno} = $topic;
    } else {
	&error("Unknown document id `$docno' for Trials run");
	next;
    }
    $numret{$topic}++;
	
}


# Do global checks:
#   error if some topic has no (or too many) documents retrieved for it
#   warn if too few documents retrieved for a topic
foreach $t (@topics) {
    if ($numret{$t} == 0) {
        &error("No documents retrieved for topic $t");
    }
    elsif ($numret{$t} > $MAX_RET) {
        &error("Too many documents ($numret{$t}) retrieved for topic $t");
    }
}


print ERRLOG "Finished processing $results_file\n";
close ERRLOG || die "Close failed for error log $errlog: $!\n";
if ($num_errors) {
    exit 255;
}
exit 0;


# print error message, keeping track of total number of errors
sub error {
    my $msg_string = pop(@_);

    print ERRLOG 
	"$0 of $results_file: Error on line $line_num --- $msg_string\n";

    $num_errors++;
    if ($num_errors > $MAX_ERRORS) {
        print ERRLOG "$0 of $results_file: Quit. Too many errors!\n";
        close ERRLOG ||
	    die "Close failed for error log $errlog: $!\n";
	exit 255;
    }
}


# Check for a valid docid for this type of run
#
sub check_docno {
    my ($docno) = @_;

    return ($docno =~ /^NCT[0-9]+$/); 
}

