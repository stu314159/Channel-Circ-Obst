%ch_circ_obst3D_genInput.m


clear
clc
close('all')

lattice_selection = 1;
% 1 = D3Q15

dynamics = 1;
% 1 = LBGK

entropic = 0;
% 0 = no

initialization = 0;
% 0 = initialize fIn to zero speed

Num_ts = 10000;
ts_rep_freq = 1000;
plot_freq = 1000000;

% overall domain dimensions
Lx_p = 1;
Ly_p = 1;
Lz_p = 5;

obstruction_type = 0;

Re = 100;
Ny_divs = 40;
dt = 2.5e-3;

% cylinder

switch obstruction_type 
    
    case 0
        Lo=Ly_p;
        R=0;x_c=0;z_c=0;
    
    case 1
       R = Ly_p/10;
       Lo = 2*R;
       x_c = Lx_p/2;
       %y_c = Ly_p/2;
       z_c = Lz_p/3;
        
        
end



fluid = 2;
% 1 = glycerin
% 2 = glycol
% 3 = water
% 4 = fake fluid for benchmarks

switch fluid
    case 1
        rho_p = 1260;
        nu_p = 1.49/rho_p;
        
    case 2
        rho_p = 965.3;
        nu_p = 0.06/rho_p;
        
    case 3
        rho_p = 1000;
        nu_p = 1e-3/rho_p;
        
    case 4
        rho_p = 1000;
        nu_p = 0.01;
        
end


% non-dimensionalization
Uo = nu_p*Re/Lo;
To = Lo/Uo;
Uavg = Uo;

Ld = 1; Td = 1; Ud = (To/Lo)*Uavg;
nu_d = 1/Re;

dx = 1/(Ny_divs-1);
u_lbm = (dt/dx)*Ud;
nu_lbm = (dt/(dx^2))*nu_d;
omega = 1/(3*nu_lbm+(1/2));

u_conv_fact = (dt/dx)*(To/Lo);
t_conv_fact = (dt*To);
l_conv_fact = dx*Lo;
p_conv_fact = ((l_conv_fact/t_conv_fact)^2)*(1/3);

rho_lbm = rho_p;

Ny = ceil((Ny_divs-1)*(Ly_p/Lo))+1;
Nx = ceil((Ny_divs-1)*(Lx_p/Lo))+1;
Nz = ceil((Ny_divs-1)*(Lz_p/Lo))+1;

nnodes = Nx*Ny*Nz;

fprintf('Number of Lattice-points = %d.\n',nnodes);
fprintf('Number of time-steps = %d. \n',Num_ts);
%fprintf('Predicted execution time = %g.\n', predicted_ex_time);

fprintf('LBM viscosity = %g. \n',nu_lbm);
fprintf('LBM relaxation parameter (omega) = %g. \n',omega);
fprintf('LBM flow Mach number = %g. \n',u_lbm);

input_string = sprintf('Do you wish to continue? [Y/n] \n');

run_dec = input(input_string,'s');

if ((run_dec ~= 'n') && (run_dec ~= 'N'))
    
    fprintf('Ok! Cross your fingers!! \n');
    
   % s = system('rm *.lbm');
    params = fopen('params.lbm','w');
    fprintf(params,'%d \n',lattice_selection);
    fprintf(params,'%d \n',dynamics);
    fprintf(params,'%d \n',entropic);
    fprintf(params,'%d \n',initialization);
    fprintf(params,'%d \n',Num_ts);
    fprintf(params,'%d \n',ts_rep_freq);
    fprintf(params,'%d \n',plot_freq);
    fprintf(params,'%d \n',obstruction_type);% obst type
    fprintf(params,'%f \n',x_c);% obst param 1
    fprintf(params,'%f \n',z_c); %obst param 2
    fprintf(params,'%f \n',R);% obst param 3
    fprintf(params,'%f \n',0);% obst param4
    fprintf(params,'%f \n',rho_lbm);
    fprintf(params,'%f \n',u_lbm);
    fprintf(params,'%f \n',omega);
    fprintf(params,'%d \n',Nx);
    fprintf(params,'%d \n',Ny);
    fprintf(params,'%d \n',Nz);
    fprintf(params,'%f \n',t_conv_fact);
    fprintf(params,'%f \n',l_conv_fact);
    
    fclose(params);
else
    fprintf('Run aborted.  Better luck next time!\n');
end


