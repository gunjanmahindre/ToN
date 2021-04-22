#!/bin/bash
#normal_matrix_completion.py
#0 , 5, 10, 20, 40, 60, 80
#fraction = 0 #percentage of elements to be removed
#np.savetxt('nw_0.txt', dc, delimiter = '\t')
#run 7 times
#mean_error_ton.py
#Iteration_1
run=100
j=0
y=1
rm -rf tpm_total.txt
#rm -rf dev.txt
for x in $(eval echo "{1..$run}")
do
	for i in 0 10 20 40 60 80
	do
		echo "---------------------------------------------------------------------------------Iteration running is $x"
		echo "--------------------------------------------------------------------------------Percentage running is $i"
		sed -i "s|fraction = ${j}|fraction = ${i}|g" generate_TPMs.py
		sed -i "s|nw_${j}.txt|nw_${i}.txt|g" generate_TPMs.py
		#run script1
		~/anaconda3/bin/python3.6 generate_TPMs.py
		j=$i
		if [ $i == 80 ]
		then
			sed -i "s|fraction = ${i}|fraction = 0|g" generate_TPMs.py
			sed -i "s|nw_${i}.txt|nw_0.txt|g" generate_TPMs.py
		fi
	done

	echo "---------------------------------------------------------------------------------------Iteration script running is $x" 
	sed -i "s|Iteration_${y}|Iteration_${x}|g" TPM_error_percentage_call.py
	#run script2
	~/anaconda3/bin/python3.6 TPM_error_percentage_call.py
	y=$x
	if [ $x == $run ]
	then
		sed -i "s|Iteration_${x}|Iteration_1|g" TPM_error_percentage_call.py
	fi
	echo "----------------------------------"
done

# Calculate average
FILE=TPM_averaged.txt
for a in 0 10 20 40 60 80
do
	c=`cat tpm_total.txt | grep "$a% " -c`
	total=0
	for b in $(eval echo "{1..$c}")
	do
		value=`cat tpm_total.txt | grep "$a% " -m$b | tail -n1 | rev | cut -d' ' -f1 | rev`
		total=`echo $total + $value | bc`
	done
	total=`echo $total / $c | bc -l`
	echo "$a%     $total" >> $FILE
done

#FILE2=dev_average_hourglass.txt
#for a in 0 5 10 20 40 60 80
#do
#        c=`cat dev.txt | grep "$a% " -c`
#        total=0
#        for b in $(eval echo "{1..$c}")
#        do
#                value=`cat dev.txt | grep "$a% " -m$b | tail -n1 | rev | cut -d' ' -f1 | rev`
#                total=`echo $total + $value | bc`
#        done
#        total=`echo $total / $c | bc -l`
#        echo "$a%     $total" >> $FILE2
#done
