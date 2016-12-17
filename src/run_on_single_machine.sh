#!/bin/bash

handler()
{
	echo "Killed"
	# kill all background processes
	pkill -f "python $1"
}

trap handler SIGINT

if [ "$1" = "" ] || [ "$2" = "" ] || [ "$3" = "" ] || [ "$4" = "" ]; then
	echo "usage: $0 <source_file> <param_servers_number> <workers_number> \
	<port_number_start>"
	echo "IMPORTANT: don't run this script with sh"
	exit 1
fi
servNum=$2
workNum=$3

# same ip different ports
log_dir=/home/$USER/tmp_logs_rc
rm  -r $log_dir


# initialize addresses and ports
pservers="localhost:$4"
st=$4
port=$((st+1))
for i in $(seq 1 $((servNum-1))); do
	pservers=$pservers",localhost:$port"
	port=$((port+1))
done
echo $pservers

workers="localhost:$port"
port=$((port+1))
for i in $(seq 1 $((workNum-1))); do
	workers=$workers",localhost:$port"
	port=$((port+1))
done
echo $workers

# start parameter servers
for i in $(seq 0 $((servNum-1))); do
	python $1 --ps_hosts=$pservers --worker_hosts=$workers \
		--job_name=ps --task_index=$i --log_dir=$log_dir &
done

# start workers
for i in $(seq 0 $((workNum-1))); do
	python $1 --ps_hosts=$pservers --worker_hosts=$workers \
		--job_name=worker --task_index=$i --log_dir=$log_dir &
#	if [ $i == 0 ]; then
#		sleep 2
#	fi
done

# wait for the last worker to finish
wait $!
pkill -f "python $1"
