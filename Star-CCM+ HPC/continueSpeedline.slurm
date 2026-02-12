#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH -J "C42-7B CON"     # Job Name
#SBATCH --time=12:00:00   # walltime (hh:mm:ss)
#SBATCH -C 'm12'   # features syntax (use quotes): -C 'a&b&c&d'
#SBATCH --ntasks=128   # number of processor cores 
#SBATCH --nodes=1 
#SBATCH --mem-per-cpu=1G   # memory per CPU
#SBATCH --mail-user=oliphanc@byu.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --licenses=starccm_ccmpsuite:1



###############################################################
##### Should not need to change anything below this point #####
###############################################################
 
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

#Edit this line with a new .sim file name vvv
starccm+ -np $SLURM_NTASKS -machinefile $machinefile -batch continueSpeedline.java 0.16642.sim
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

printf "____\n"
printf "Ending starccm+ run at "
date
printf "\n____\n"

exit 0
