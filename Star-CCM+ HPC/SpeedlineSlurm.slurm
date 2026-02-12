#!/bin/bash

#Submit this script with: sbatch thefilename


#SBATCH --time=72:00:00   # walltime (hh:mm:ss) -- Should be set to #Points*4 (usually 24 hours)
#SBATCH -C 'm12'   # features syntax (use quotes): -C 'a&b&c&d'
#SBATCH --ntasks=128   # number of processor cores 
#SBATCH --nodes=1 
#SBATCH --mem-per-cpu=2G # memory per CPU
#SBATCH --mail-user=oliphanc@byu.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --licenses=starccm_ccmpsuite:1


##########################################################
##### Do not need to change anything in this section #####
##########################################################
 
# Set the output permissions on files written from my script to be 0660 and directories to be 0770
umask 0007


# Generate machinefile
machinefile=$(/apps/orcutils/generate_mpich_machinefile)
#machinefile=$(./mpich_machinefile)


export SPEEDLINERPM=$1
startfile=$2
counter=0

echo $SPEEDLINERPM
echo $startfile
echo $SLURM_JOB_NAME

###cd $SIM_PATH
printf "Job debugging info\n____\n"
printf "Machinefile:\n"
cat $machinefile
echo "----"
printf "____\n"
printf "SIM_FILE variable: $startfile\n"
printf "____\n"
printf "Current Directory: $PWD\n"
printf "____\n"
printf "Starting starccm+ run at "
date
printf "\n____\n"

module purge
module load starccm+/2206

ulimit -a


cd $SPEEDLINERPM
echo "Running Speedline at "$SPEEDLINERPM
starccm+ -np $SLURM_NTASKS -machinefile $machinefile -batch ../speedline_up.java $startfile
starccm+ -np $SLURM_NTASKS -machinefile $machinefile -batch ../speedline_down.java $startfile
if test -f .SUCCESS
then
	rm *@*.sim
	tar -cvf $SPEEDLINERPM.tar *.sim --remove-files
	zstd $SPEEDLINERPM.tar --rm
	rm .SUCCESS
else
	echo "Failed on "$SPEEDLINERPM
	counter=$((counter+1))
fi

rm "$machinefile"

chgrp -R fslg_Concepts_Runs *.*
chmod -R 770 *.*

printf "____\n"
printf "Ending starccm+ run at "
date
printf "\n____\n"

exit $counter
