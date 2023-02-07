function [Res] = func_SSFAD(hsi,win_out,win_in,parameter)
%% Function Instruction
% Author: Zephyr Hou
% Time: 2022-02-01
%
% Function Usage:
%   [Res] = func_SSFAD(hsi,win_out,win_in)
% Inputs:
%   hsi - 3D data matrix (num_row x num_col x num_dim)
%   win_out - spatial size window of outer(e.g., 3, 5, 7, 9,...)
%   win_in - spatial size window of inner(e.g., 3, 5, 7, 9,...)
%   parameter - fusion strategy{'average' or 'adaptive'}
% Outputs:
%   Res - anomaly detection result output(num_row x num_col)  

%% ======================= Main Function ==========================
%step1. Anomaly Detector in Spectial Domain 
R1_spec=func_SpecAD(hsi, win_out, win_in); 
R1_spec=(R1_spec-min(R1_spec(:)))/(max(R1_spec(:))-min(R1_spec(:))); % normalized the result

%step2. Anomaly Detection in Spatial Domain
R2_spat = func_SpatAD(hsi, win_in);
R2_spat=(R2_spat-min(R2_spat(:)))/(max(R2_spat(:))-min(R2_spat(:)));

%step3. Fusion Strategy
% Res=(R1_spec+R2_spat)/2;
if strcmpi(parameter,'average')
    Res=(R1_spec+R2_spat)/2;
elseif strcmpi(parameter,'adaptive')
    a=norm(R1_spec)/(norm(R2_spat)+norm(R1_spec));
    b=norm(R2_spat)/(norm(R2_spat)+norm(R1_spec));
    Res=a*R1_spec+b*R2_spat;
else
    error('Fusion parameters are wrong or missing, please recheck!')
end
    
end

%% ===================== Sub-function1 ===========================
% Spectral domain anomaly detector for hyperspectral imagery
function result = func_SpecAD(Data, win_out, win_in)
%%
% Author: Zephyr Hou
% Time: 2020-09-13
%
% Usage
%   [result] = func_SWLRX(Data, window, lambda)
% Inputs
%   Data - 3D data matrix (num_row x num_col x num_dim)
%   win_out - spatial size window of outer(e.g., 3, 5, 7, 9,...)
%   win_in - spatial size window of inner(e.g., 3, 5, 7, 9,...)
%   lambda - regularization parameter
% Outputs
%   result - Detector output (num_row x num_col)  
%% Main Function
[rows, cols, bands] = size(Data);
result = zeros(rows, cols);
t = fix(win_out/2);
t1 = fix(win_in/2);
M = win_out^2;

% padding avoid edges
DataTest = zeros(rows+2*t,cols+2*t, bands);
DataTest(t+1:rows+t, t+1:cols+t, :) = Data;
DataTest(t+1:rows+t, 1:t, :) = Data(:, t:-1:1, :);
DataTest(t+1:rows+t, t+cols+1:cols+2*t, :) = Data(:, cols:-1:cols-t+1, :);
DataTest(1:t, :, :) = DataTest(2*t:-1:(t+1), :, :);
DataTest(t+rows+1:rows+2*t, :, :) = DataTest(t+rows:-1:(rows+1), :, :);
%% IDW
tempWinX=zeros(win_out,win_out);
tempWinY=zeros(win_out,win_out);
for i=1:win_out
    tempWinX(i,:)=-t:t;
    tempWinY(:,i)=t:-1:-t;
end

Dd=sqrt(tempWinX.^2+tempWinY.^2);
Dd(t-t1+1:t+t1+1, t-t1+1:t+t1+1) = NaN;
IDW = reshape(Dd, M, 1);
IDW(isnan(IDW(:,1)),:) = [];
SumW=sum(IDW.^-2);
IDW=IDW.^-2/SumW;

for i = t+1:rows+t 
    for j = t+1:cols+t
        block = DataTest(i-t: i+t, j-t: j+t, :);
        %local median-mean line (LMML) metric backrgound rectified
        [h,l,bands]=size(block);
        dataset= reshape(block,h*l,bands)'; % bands x rc
        meanVal = mean(dataset')'; % bands x 1

        Mtemp=zeros(1,h*l);
        for ii=1:h*l
            Mtemp(1,ii)=norm(dataset(:,ii)-meanVal);
        end
        [~,ind]=sort(Mtemp,'ascend');

        if mod(h*l,2)==0  % even
            medianVal=(dataset(:,ind(h*l/2))+dataset(:,ind(h*l/2+1)))/2; % bands x 1
        else  % odd
            medianVal=dataset(:,ind((h*l+1)/2));   % bands x 1
        end

        paraV=zeros(1,h*l);
        dataset_mml=zeros(bands,h*l);  % bands x rc
        for ii=1:h*l
            paraV(1,ii)=((dataset(:,ii)-medianVal)'*(meanVal-medianVal))/((meanVal-medianVal)'*(meanVal-medianVal));
            dataset_mml(:,ii)=(1-paraV(1,ii))*medianVal+paraV(1,ii)*meanVal;
        end

        block=reshape(dataset_mml',h,l,bands);   % the LMML-Rectified dataset
        
        y = squeeze(block(ceil(h/2),ceil(l/2),:)).';
        LocPixs=block(ceil(h/2)-t1:ceil(h/2)+t1,ceil(l/2)-t1:ceil(l/2)+t1,:);

        block(t-t1+1:t+t1+1, t-t1+1:t+t1+1, :) = NaN;   % 去掉内窗像元
        block = reshape(block, M, bands);
        block(isnan(block(:, 1)), :) = [];
        BacVec = block'; %  num_dim x num_sam
              
        for k=1:size(BacVec,2)
            BacVec(:,k)=(1-exp(-0.5*norm(y'-BacVec(:,k).*IDW(k))))*(y'-BacVec(:,k).*IDW(k));   % 这一步有效果
        end      
        Sigma = (BacVec* BacVec');
        Sigma_inv = pinv(Sigma); 
        
       % Saliency Weight Calculation (inner window)
  
        LocPixs=reshape(LocPixs,win_in*win_in,bands)';       % bands x num_in 
        % spectral angle
        temp=repmat(y',[1,size(LocPixs,2)]);                 % bands x num_in     
        fz=sum(temp.*LocPixs);                               % 1 x num_in
        fm=sqrt(sum(temp.^2).*sum(LocPixs.^2));              % 1 x num_in
        dis_Spectrum=acos(fz./fm);                           
        
        % 位置距离(坐标之间的欧式距离)
        tempWinX=zeros(win_in,win_in);
        tempWinY=zeros(win_in,win_in);
        for ii=1:win_in
            tempWinX(ii,:)=-t1:t1;
            tempWinY(:,ii)=t1:-1:-t1;
        end
        dis_Position=sqrt(tempWinX.^2+tempWinY.^2);        % num_in x mum_in
        dis_Pos=reshape(dis_Position,win_in*win_in,1)';    % 1 x num_in
       
        Const=1; % 常量设置为1
        dis_Sal=dis_Spectrum./(1+Const*dis_Pos);           % 1 x num_in 
        dis_Sal_ava=sum(dis_Sal)/(size(LocPixs,2)-1);      % constant
        
        result(i-t,j-t)=(y*Sigma_inv*y')*dis_Sal_ava;  
    end
end

end


%% ===================== Sub-function2 ===========================
% Spatial domain anomaly detector for hyperspectral imagery
function result = func_SpatAD(Data, win_in)
% Compiled by ZephyrHou on 2020-09-14
%
% Function Usage:
%   [result] = func_SpatAD(Data, window, lambda)
% Inputs:
%   Data - 3D data matrix (num_row x num_col x num_dim)
%   win_out - spatial size window of outer(e.g., 3, 5, 7, 9,...)
%   win_in - spatial size window of inner(e.g., 3, 5, 7, 9,...)
%   lambda - regularization parameter
% Outputs:
%   result - Detector output (num_row x num_col)  

%% Main Function
[rows,cols,bands] = size(Data);
result = zeros(rows, cols);
win_out=win_in*3;
t = fix(win_out/2);
t1 = fix(win_in/2);
M = win_out^2;

% padding avoid edges 
DataTest = zeros(rows+2*t,cols+2*t, bands);
DataTest(t+1:rows+t, t+1:cols+t, :) = Data;
DataTest(t+1:rows+t, 1:t, :) = Data(:, t:-1:1, :);
DataTest(t+1:rows+t, t+cols+1:cols+2*t, :) = Data(:, cols:-1:cols-t+1, :);
DataTest(1:t, :, :) = DataTest(2*t:-1:(t+1), :, :);
DataTest(t+rows+1:rows+2*t, :, :) = DataTest(t+rows:-1:(rows+1), :, :);

Img=DataTest; 
for i = t+1:rows+t 
    for j = t+1:cols+t
        block = Img(i-t:i+t,j-t:j+t,:);             % win_out x win_out x bands   
        WinIn=Img(i-t1:i+t1,j-t1:j+t1,:);           % win_in x win_in x bands
        yMat=reshape(WinIn,win_in*win_in,bands)';    
        
        Res=[];       

        for k=t1+1:win_out-t1
            BacMat=block(1:win_in,k-t1:k+t1,:);
            BacMat=reshape(BacMat,win_in*win_in,bands)';        
            temp=trace((yMat-BacMat)'*(yMat-BacMat));
            Res=[Res,temp];
         %%    
            BacMat3=block(2*win_in+1:3*win_in,k-t1:k+t1,:);
            BacMat3=reshape(BacMat3,win_in*win_in,bands)';     
            temp=trace((yMat-BacMat3)'*(yMat-BacMat3));
            Res=[Res,temp];
        end
        
        for k=t1+2:win_out-t1-1
            BacMat4=block(k-t1:k+t1,1:win_in,:); 
            BacMat4=reshape(BacMat4,win_in*win_in,bands)';            
            temp=trace((yMat-BacMat4)'*(yMat-BacMat4));
            Res=[Res,temp];
            %%
            BacMat2=block(k-t1:k+t1,2*win_in+1:3*win_in,:); 
            BacMat2=reshape(BacMat2,win_in*win_in,bands)';            
            temp=trace((yMat-BacMat2)'*(yMat-BacMat2));
            Res=[Res,temp];
        end                
        result(i-t,j-t)=min(Res);
    end
end
end

