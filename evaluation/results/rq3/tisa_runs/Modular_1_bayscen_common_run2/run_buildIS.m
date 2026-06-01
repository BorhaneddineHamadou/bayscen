addpath('C:/Users/BH280005/Desktop/BayScen/tisa/tisa_matlab/ISA-code/InstanceSpace');
try
    buildIS('C:/Users/BH280005/Desktop/BayScen/results/rq3/tisa_runs/Modular_1_bayscen_common_run2/','C:/Users/BH280005/Desktop/BayScen/tisa/tisa_matlab/ISA-code/');
    disp('EOF:SUCCESS');
catch ME
    disp('EOF:ERROR'); disp(ME.message);
end
exit;
