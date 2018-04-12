for i in 100 1000
do
   for j in 100 1000
   do
       echo "multinomial " "samples:"  $i " features:" $j 
       python generate.py simulate_poisson \
           --output_dir ../data/poisson_"s$i"_"f$j" \
           --num_samples $i --num_features $j \
           --low -3 --high 3 --alpha_mean -3 --kappa_mean -1
   done
done