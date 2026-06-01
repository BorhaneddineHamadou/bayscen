addpath('C:/Users/BH280005/Desktop/BayScen/tisa/tisa_matlab/ISA-code/InstanceSpace');
try
    buildIS('C:/Users/BH280005/Desktop/BayScen/results/rq3/tisa_runs/Interfuser_2_random_run3/','C:/Users/BH280005/Desktop/BayScen/tisa/tisa_matlab/ISA-code/');
    disp('EOF:SUCCESS');
catch ME
    disp('EOF:ERROR'); disp(ME.message);
end
exit;
