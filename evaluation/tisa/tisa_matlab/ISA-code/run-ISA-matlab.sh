#!/bin/bash

rootdir='path-of-samples'
resultsdir='path-of-result-directory-with-csv-to-safe-TISA-metrics'
cd /home/nnee0002/Documents/MATILDA/InstanceSpace
for i in {1..30}
do
echo $i
	workingDirectory=$rootdir$i'/'
	echo 'path for log file: ' $workingDirectory;
	outputfile=$workingDirectory'matilda_logs.txt'
	echo 'path for output: ' $outputfile;
	/home/user/Downloads/Matlab-installation/bin/matlab -nodisplay -nosplash -nodesktop -r "buildIS('$workingDirectory','$resultsdir'); exit;" > ${outputfile} 2>&1
done
