for i in 100 1000 
do
   for j in 1000 5000
   do
       echo "multinomial " "samples:"  $i " features:" $j 
       python generate.py simulate_poisson \
           --output_dir ../data/poisson_"s$i"_"f$j" \
          --num_samples $i --num_features $j \
          --low -5 --high 5 --alpha_mean 4
   done
done