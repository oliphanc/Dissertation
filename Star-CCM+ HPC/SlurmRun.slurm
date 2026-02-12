#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH -J "C5-18B"     # Job Name
#SBATCH --time=72:00:00   # walltime (hh:mm:ss) -- Should be set to #Speedlines*5
#SBATCH -C 'm12'   # features syntax (use quotes): -C 'a&b&c&d'
#SBATCH --ntasks=128   # number of processor cores 
#SBATCH --nodes=1 
#SBATCH --mem-per-cpu=2G   # memory per CPU
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

###cd $SIM_PATH
printf "Job debugging info\n____\n"
printf "Machinefile:\n"
cat $machinefile
echo "----"
printf "____\n"
printf "SIM_FILE variable: $SIM_FILE\n"
printf "____\n"
printf "Current Directory: $PWD\n"
printf "____\n"
printf "Starting starccm+ run at "
date
printf "\n____\n"

module purge
module load starccm+/2206

ulimit -a

export counter=0


speeds=(14000 16000 18000)	#<--------- Add speeds here!

for speed in ${speeds[@]}; do
	export speed
	printf "Current speed: $speed\n"
	starccm+ -np $SLURM_NTASKS -machinefile $machinefile -batch updateStartingConditions.java start.sim
	if test -f .SIMNAME
	then
		read startfile < .SIMNAME 
		printf "Speedline start file: $startfile\n"
		sbatch -J "${SLURM_JOB_NAME}${speed}" SpeedlineSlurm.slurm $speed $startfile
		rm .SIMNAME
	else
		echo "Failed to submit job at $speed RPM"
	fi
done


rm "$machinefile"

chgrp fslg_Concepts_Runs *.*

printf "____\n"
printf "Ending starccm+ run at "
date
printf "\n____\n"

exit $counter
