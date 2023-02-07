%% SSFAD
% Paper: ¡¶A Spectral Spatial Fusion Anomaly Detection Method for Hyperspectral Imagery¡·
% Author: Zephyr Hou
% Compiled Time:2022-02-02
%%
clc;clear;close all;
%% Load Hyperspectral Datasets
[imgname,pathname]=uigetfile('*.*', 'Select the original image file','./Datasets');
if imgname == 0
    error("There is no file selected !")
else
    str=strcat(pathname,imgname);
    [pathstr,name,ext]=fileparts(str);
    if ext == '.mat'
        disp('The hyperspectral dataset is :')
        disp(str);
        addpath(pathname);
        load(strcat(pathname,imgname));
    end
end
%% =======================================================================================
% Automatic image name recognition
[pathstr,name,ext]=fileparts(str);
name=strrep(name,'_','\_');

[rows,cols,bands]=size(hsi);
label_value=reshape(hsi_gt,1,rows*cols);

%% Optimal Parameters of Different Methods
%Proposed(win_out,win_in),LRX(win_out,win_in),CRD(win_out,win_in,lambda),RPCA(lambda),GTVLRR(lambda,beta,gamma)
OptPara={   'Methods',  'SSFAD';  
       'Gulfport[A4]',  [5, 3]; 
   'Pavia Centra[B4]',  [5, 3]; 
    'Gainesville[U3]',  [5, 3];
                'Cir',  [17,15]};
%% Name of Experimental dataset(hsi,hsi_gt)
if strcmp(name,'abu-airport-4')
    ExperimentData='Gulfport[A4]';   
elseif strcmp(name,'abu-beach-3')
    ExperimentData='Champagne Bay[B3]'; 
elseif strcmp(name,'abu-beach-4') 
    ExperimentData='Pavia Centra[B4]';
elseif strcmp(name,'abu-urban-3')
    ExperimentData='Gainesville[U3]';  
elseif strcmp(name,'Cri') 
    ExperimentData='Cir';   
else
    warning('Unexpected dataset, please reselect the dataset or modify the parameters in different algorithms!')   
end

%%
ImgName=OptPara(:,1);
NameVec=strcmp(ImgName,ExperimentData);
ind_row=find(NameVec==1);

Proposed='SSFAD';

MethodName=OptPara(1,:);

ind_pro=find(strcmp(MethodName,Proposed)==1);
para_my=OptPara{ind_row,ind_pro};

%% ==========================Contrast Experiment==============================
%% Proposed SSFAD
disp('Running Proposed SSFAD, Please wait...')
tic
R0 = func_SSFAD(hsi,para_my(1),para_my(2),'adaptive'); 
t0=toc;
disp(['SSFAD Time£º',num2str(t0)])
R0value = reshape(R0,1,rows*cols);
[FA0,PD0] = perfcurve(label_value,R0value,'1') ;
AUC0=-sum((FA0(1:end-1)-FA0(2:end)).*(PD0(2:end)+PD0(1:end-1))/2);

disp('---------------------------------------------------------------------')
disp('SSFAD')
disp(['AUC:     ',num2str(AUC0),'          Time:     ',num2str(t0)])
disp('-------------------------------------------------------------------')
%% ROC Curves Display

figure;
plot(FA0, PD0, 'k-', 'LineWidth', 2);  hold on
xlabel('False alarm rate'); ylabel('Probability of detection');
legend('SSFAD','location','southeast')

