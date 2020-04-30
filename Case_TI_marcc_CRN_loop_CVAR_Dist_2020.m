%% No Free Lunch: Maximizing E[SR]
%% !!! Problem settings
%  1. Learning by doing- you know better about the performance of GI(i) in the second stage, 
%     if the investments on GI(i) are higher than some tresholds.
%  2. Scenario: randomly generate m1 scenario for stage II realization
%  3. If no enough investment on GI(i), the performance will remain uncertain
%     with the same deistribution in stage I( we did not learn)
%  4.y(i): the binary variables are relaxed in this model 0<=y(i)<=1
% sample mean is adjusted to mu
% sample variance is adjusted to sigma^2
%% parameters
% CVAR is included in the 2nd-stage to make sure that we get the desirable outcome. 
clear all; clc; 
nGI = 5; %RG, IT, PP, RB, RG
nsub= 3; %s1,s2,s3
n1=nGI*nsub;                % the no. of r.v.:RG* 3, IT*3,...
r1=3;                % the number of learning states: learning nothing, partial info. & perfect info
% Number of scenairos in stage 1 & 2
m1=200;        % the number of scenarios in Stage 2
m2=10;          % the number of scenarios in scenario j, stage 2, for uncertain case. 
% m1 should to be relatively large or the model will take advantage of the samples.
T1=5;T2=20; % the time horizon of period 1 and 2.; 
% total 25 years, T1 and T2 starts at year 1 and year 6, respectively 
Budget=1.8*10^8;  %total budget 10 M/yr% 
adj1 = [1,0.95,0.9]; % cost adjustment factor for subsewersheds
CVAR= 1*10^6;  % stormwater reduction (m3)
M=1*10^7;            % an arbitrary big number
% Annualized Costs ($/m2/yr -installation)
Ecost = [26.6,27.9,12.3,27.5,21.3];  % average costs
EcostH = [34.6,36.3,14.8,28.9,25.5]; % high costs
EcostL = [18.6,19.5,9.9,26.2,17.0];  % low costs
% Ecost = [10,9,8.3,23.5,15.4];  % average costs
% EcostH = [11,10.5,8.7,24.7,18.5]; % high costs
% EcostL = [9,7.5,7.9,22.3,12.3];  % low costs
EcostALL = Ecost.*adj1';  % cost matrix
EcostHALL = EcostH.*adj1';
EcostLALL = EcostL.*adj1';
EcostALL = EcostALL(:)';
EcostHALL = EcostHALL(:)';
EcostLALL = EcostLALL(:)';

% Deterioration
D2_a = [0.7,0.6,0.5,0.7,0.9];
D2_b = [0.9,0.9,0.7,0.9,0.95];
D1_a = [0.9,0.9,0.7,0.9,0.95];
D1_b = [1  ,1  ,1  ,1  ,1   ];

% D2_a = [0.99,0.99,0.99,0.99,0.99];
% D2_b = [1  ,1  ,1  ,1  ,1  ];
% D1_a = [0.99,0.99,0.99,0.99,0.99];
% D1_b = [1  ,1  ,1  ,1  ,1   ];
% Learning Thresholds
% Lt1 = repmat(Ecost*10^4*2,3,1);
% Lt2 = repmat(Ecost*10^4*6,3,1);
% Lt1 = repmat(Ecost.*[9,5,2,1,2],3,1)*10000;
% Lt2 = repmat(Ecost.*[40,20,3,2,4],3,1)*10000;
% LT1=Lt1(:)';
% LT2=Lt2(:)';
% LT1 = [24,24,24,14,14,14,2,2,2,1,1,1,4,4,4]*10^4;
% LT2 = [100,100,100,50,50,50,4,4,4,2,2,2,9,9,9]*10^4;
LT1 = [2200,2200,2200,1000,1000,1000,160,160,160,30,30,30,350,350,350]*10^3;
LT2 = [10000,10000,10000,4000,4000,4000,350,350,350,60,60,60,700,700,700]*10^3;

% three SMPs and their performance follow normal dist. w/ means and
% variance as below.
% mu=[RGs1,RGs2,RGs3,ITs1,...,RBs1,...] (m3/$/yr)
% sigma = [RGs1,RGs2,RGs3,ITs1,...,RBs1,...] (m3/$/yr)
% giP1=[1.13,1.62,0.19,1.09,0.03];
% giSTD=[0.18,0.32,0.04,0.11,0.01]; % Original Number
% giP1=[0.437,0.537,0.127,0.927,0.024];
% giP1=[0.437,0.537,0.127,0.4,0.024];
% giSTD=[0.106,0.127,0.036,0.087,0.004];  % Test Number

giP1=[0.437,0.537,0.127,0.4,0.024];
giSTD=[0.106,0.127,0.036,0.087,0.004];  % Test Number

giP_all =giP1./adj1';
giSTD_all= giSTD./adj1';

% mean and standard devation of GI's performance (m3/$/yr)
mu = giP_all(:)';
sigma = giSTD_all(:)';
TA = [24,30,1.5,108,1];% Treated area ratio 

alpha=0.1;

%%%%% Investment Upper Bound 
Imp_A = [458.4,410.2,274.3]*10^4;
Roof_A = Imp_A*0.4;    % inpervious roof area
Ground_A = Imp_A*0.6;  % impervious ground area
G_R_A = repmat([Ground_A;Roof_A], [m1,1]);  % imperfivous G/R area 

ub = zeros(1,n1);
for i= 1:nGI
    if i < 4
         for j = 1:nsub
             ub(1,((i-1)*3+j)) = Ground_A(j)/TA(i)*(Ecost(i)*adj1(j));
         end
    else 
         for j = 1:nsub
             ub(1,((i-1)*3+j)) = Roof_A(j)/TA(i)*(Ecost(i)*adj1(j));
         end    
    end
end

%%%% 
ndv = n1+n1*m1+n1*m1+n1*m1+n1*r1+1+m1*m2;  
% the number of d.v. - x1(n1)+x2(n1*m1)+x2_FL(n1*m1)+x2_PL(n1*m1)+L(n1*r1)+rho(1)+z(m1*m2)
nst = m1+n1*m1+m1*2*nsub+(n1*m1*3)+n1*5+m1*m2+1;       % no. of constraints
% m1 budget const., 
% n1*m1 UB const.
% 2*nsub*m1 Groudn/roof area consts.
% n1*m1*r1 big M const:  for x2, x2_FL and x2_PL ,
% n1*5 learning const:
% m1*m2+1 CVaR const
p1=1/m1;                % the probability of scenario j in stage 2
p2=1/m2;             % the probability of scenario jk in stage 2
% the variance reduction rate for Basic/Advance Learning cases 
beta1 = repmat([0.7,0.7,0.7,0.7,0.7],[3,1]);
beta2 = repmat([0.5,0.5,0.5,0.5,0.5],[3,1]);
Beta1 = beta1(:)';    
Beta2 = beta2(:)';
% increasement multiplier of Technology Improvement
% gamma1 = ones([3,nGI]);
% gamma2 = gamma1;

gamma1 = repmat([1.1,1.1,1.1,1.05,1.1],[3,1]);
gamma2 = repmat([1.3,1.2,1.2,1.1,1.3],[3,1]);
Gamma1=gamma1(:)';    
Gamma2=gamma2(:)';
%% generate stage 2 scenarios j in J (m1)
% Stage 1 performances realizations
% standard error = sigma/sqrt(n), e.g. sigma= 0.2, n =100, Std_err=0.02
% invest in stage 1 and 2 both would reduce the uncertainty (sd) to
% standard error which depends on the sampple size of stage 1 scenario (m1)

% read samples from excel
[samples,txt] = xlsread('GIp_samples.xlsx');

s1=zeros(n1,m1); % no learning
    for i = 1:nGI % for each GI, sample from the data, and adjust for the cost difference
        for j = 1:nsub
            s1((i-1)*nsub+j,:) = datasample(samples(:,i),m1)/adj1(j) ;  
        end
    end 
s1=(sigma./std(s1',1)).*s1'+repmat(mu-mean(s1').*(sigma./std(s1',1)),[m1,1]);

% generate deterioration rates for stage 1 implementation
d1=zeros(n1,m1); % stage 1 deterioration rate
rand_no = rand(n1,m1);   %
    for i = 1:nGI % for each GI, sample from the data, and adjust for the cost difference
        d1((i-1)*nsub+(1:nsub),:) = rand_no((i-1)*nsub+(1:nsub),:)*(D1_b(i)-D1_a(i))+D1_a(i) ;  
    end
sigma_d1 = (D1_b-D1_a)/(12^0.5);
temp = repmat(sigma_d1,[3,1]);
sigma_d1=temp(:)';
mean_d1 = (D1_a+D1_b)/2;
temp = repmat(mean_d1,[3,1]);
mean_d1 = temp(:)';
d11=(sigma_d1./std(d1',1)).*d1'+repmat(mean_d1-mean(d1').*(sigma_d1./std(d1',1)),[m1,1]);

d2=zeros(n1,m1); % stage II deterioration rate for Stage 1 installation - m1 samples
    for i = 1:nGI % for each GI, sample from the data, and adjust for the cost difference
        d2((i-1)*nsub+(1:nsub),:) = rand(nsub,m1)*(D2_b(i)-D2_a(i))+D2_a(i) ;  
    end 
sigma_d2 = (D2_b-D2_a)/(12^0.5);
temp = repmat(sigma_d2,[3,1]);
sigma_d2=temp(:)';
mean_d2 = (D2_a+D2_b)/2;
temp = repmat(mean_d2,[3,1]);
mean_d2=temp(:)';
d22=(sigma_d2./std(d2',1)).*d2'+repmat(mean_d2-mean(d2').*(sigma_d2./std(d2',1)),[1,1]);

s12 = s1'.*d11'.*d22'; % this sample set only for CVAR calculation

s1=s1'.*d11';  % performance coefficient for stage 1 installation (m3/$/yr)

% Deteriration rate for stage II 
d2=zeros(n1,m2); % stage II deterioration rate - m2 samples
    for i = 1:nGI % for each GI, sample from the data, and adjust for the cost difference
        d2((i-1)*nsub+(1:nsub),:) = rand(nsub,m2)*(D2_b(i)-D2_a(i))+D2_a(i) ;  
    end 
sigma_d2 = (D2_b-D2_a)/(12^0.5);
temp = repmat(sigma_d2,[3,1]);
sigma_d2=temp(:)';
mean_d2 = (D2_a+D2_b)/2;
temp = repmat(mean_d2,[3,1]);
mean_d2=temp(:)';
d21=(sigma_d2./std(d2',1)).*d2'+repmat(mean_d2-mean(d2').*(sigma_d2./std(d2',1)),[m2,1]);

%% Basic Learning - adust the means that the sample means are equal to mu
%adust the means for Technology improvement
mu2= mu.*Gamma1;
% adjust the variance for uncertainty reduction

sigma1=sigma.*Beta1;   % reduced variance
sigma2=(sigma.^2-sigma1.^2).^0.5;  
% the first stage scenarios for PL are drawn from N(mu, sigma2) because of
% law of total variance, if we know the residual uncertainties (variance) in all the stage 2 cases 
% are the same (sigma1^2)

s1p=zeros(n1,m1);
    for i = 1:n1; %generate random sample from Norm(mu,sigma2) - central limit theorem
    s1p(i,:)=normrnd(mu2(i),sigma2(i),[1,m1]);
    end 
s1p=(sigma2./std(s1p',1)).*s1p'+repmat(mu2-mean(s1p').*(sigma2./std(s1p',1)),[m1,1]);
% adjust mean to the desired value

% s1p=s1p'.*repmat(mean_d2', [1,m1]);
s1p=s1p';

s2p=zeros(n1,m1*m2); % performances realizations for partial info case(i,j-k)
s2p1 = zeros(n1,m2);
for i=1:nGI;
    ss = datasample(samples(:,i),m2)';    
    s2p1((i-1)*nsub+(1:nsub),:)= repmat(ss,[3,1]);  
      % s2p1 is the first random samples, the rest of the sample sets will
      % be generated from this set by ajusting the mean
end
   %2. adust the means and var so that th samples have the desired
   %properties
sigma_s2p1=std(s2p1,1,2);
mean_s2p1=mean(s2p1,2);

for j=1:m1;
    s2p(:,(j-1)*m2+(1:m2))=((sigma1'./sigma_s2p1)'.*s2p1')'...
        +repmat(s1p(:,j)-((sigma1'./sigma_s2p1)'.*mean_s2p1')',[1,m2]); % adjust mean and var
    s2p(:,(j-1)*m2+(1:m2)) = s2p(:,(j-1)*m2+(1:m2)).*d21';  % multiply by the deterioration rates 
end

%% Advanced Learning - adust the means that the sample means are equal to mu
%adust the means for Technology improvement
mu2= mu.*Gamma2;
% adjust the variance for uncertainty reduction

sigma1=sigma.*Beta2;   % reduced variance
sigma2=(sigma.^2-sigma1.^2).^0.5;  
% the first stage scenarios for PL are drawn from N(mu, sigma2) because of
% law of total variance, if we know the residual uncertainties (variance) in all the stage 2 cases 
% are the same (sigma1^2)

s1f=zeros(n1,m1);
    for i = 1:n1 %generate random sample from Norm(mu,sigma2) - central limit theorem
    s1f(i,:)=normrnd(mu2(i),sigma2(i),[1,m1]);
    end 
s1f=(sigma2./std(s1f',1)).*s1f'+repmat(mu2-mean(s1f').*(sigma2./std(s1f',1)),[m1,1]);
% adjust mean to the desired value
% s1f=s1f'.*repmat(mean_d2', [1,m1]);
s1f=s1f';

s2f=zeros(n1,m1*m2); % performances realizations for partial info case(i,j-k)
s2f1 = zeros(n1,m2);

for i=1:nGI;
    ss = datasample(samples(:,i),m2)';    
    s2f1((i-1)*nsub+(1:nsub),:)= repmat(ss,[nsub,1]);  
      % s2f1 is the first random samples, the rest of the sample sets will
      % be generated from this set by ajusting the mean
end
   %2. adust the means and var so that th samples have the desired
   %properties
sigma_s2f1=std(s2f1,1,2);
mean_s2f1=mean(s2f1,2);

for j=1:m1;
    s2f(:,(j-1)*m2+(1:m2))=((sigma1'./sigma_s2f1)'.*s2f1')'...
        +repmat(s1f(:,j)-((sigma1'./sigma_s2f1)'.*mean_s2f1')',[1,m2]);
    s2f(:,(j-1)*m2+(1:m2)) = s2f(:,(j-1)*m2+(1:m2)).*d21';  % multiply by the deterioration rates 

end

%% No Learning 

s2n=zeros(n1,m1*m2); % performances realizations for no learning  cases
s2n1 = zeros(n1,m2);
for i=1:nGI;
    ss = datasample(samples(:,i),m2)';    
    s2n1((i-1)*nsub+(1:nsub),:)= repmat(ss,[3,1]);  
      % s2n1 is the first random samples, the rest of the sample sets will
      % be generated from this set by ajusting the mean
end
% adust the means that the sample means are equal to mu
sigma_s2n1=std(s2n1,1,2);
mean_s2n1=mean(s2n1,2);
for j=1:m1;
    s2n(:,(j-1)*m2+(1:m2))=((sigma'./sigma_s2n1)'.*s2n1')'...
        +repmat(mu'-((sigma'./sigma_s2n1)'.* mean_s2n1')',[1,m2]);
    s2n(:,(j-1)*m2+(1:m2)) = s2n(:,(j-1)*m2+(1:m2)).*d21';  % multiply by the deterioration rates 

end
%% Objective and constraints 
% [ Objective is Maximizing E[Benefits]                                   ]
% [ where x1 is 1st-stage decision, x2 is the 2nd-stage decision w/o      ]
% [ learning, x2_FL is the 2nd-stage decision w/ FL, x2_PL is the 2nd     ]
% [ stage decision w/ PL and l1 is a vector                               ]
% [ with elements indicating whether learning happens at GI(i)            ]
% Objective
% ***important*** obejctive represents what the decision maker knows at
% each case

f=[-mu.*mean_d1*T1/(T1+T2)-mu.*mean_d1.*mean_d2*T2/(T1+T2),...              % StageI Reduction
    -repmat(mu.*mean_d2,[1,m1])*T2/(T1+T2)*p1,...          % Stage II Reduction NL
    -reshape(s1f.*repmat(mean_d2', [1,m1]),[1,n1*m1])*T2/(T1+T2)*p1,...              % Stage II Reduction FL
    -reshape(s1p.*repmat(mean_d2', [1,m1]),[1,n1*m1])*T2/(T1+T2)*p1,...              % Stage II Reduction PL
    zeros(1,n1*r1+1+m1*m2)];                       % auxiliary varibales: rho and zi 
%%
tol=1e-4;
% Constraints
A=zeros(nst,ndv);  
%Budget constraint: sum(x1,x2,x2_FL,x2_PL)<=Budget (m1, n1+n1*m1*3)
    A(1:m1,1:n1) = (T1+T2)*[ones(m1,n1)];        % x1: independent to scenario s
% sum(x2,x2_FL,x2_PL) for each scenario s <= Budget
    for j=1:m1
        A(j,n1+(j-1)*n1+(1:n1))= T2*ones(1,n1);  %x2n
        A(j,n1+n1*m1+(j-1)*n1+(1:n1))= T2*ones(1,n1);  %x2FL        
        A(j,n1+n1*m1*2+(j-1)*n1+(1:n1))= T2*ones(1,n1);  %x2PL        
    end   
nconst=m1;                                    % index of constraint done
%UB const.: x1(n1)+x2(n1*m1)+x2_FL(n1*m1)+x2_PL(n1*m1)<= UB(i)
   A(nconst+(1:n1*m1),1:n1)=repmat(eye(n1),[m1,1]); % x1
   for j=1:m1;
        A(nconst+(j-1)*n1+(1:n1),n1+(j-1)*n1+(1:n1))=eye(n1);  %x2
        A(nconst+(j-1)*n1+(1:n1),n1+n1*m1+(j-1)*n1+(1:n1))=eye(n1);  %x2_FL
        A(nconst+(j-1)*n1+(1:n1),n1+n1*m1*2+(j-1)*n1+(1:n1))=eye(n1);%x2_PL        
   end
nconst=nconst+n1*m1;

% n1*m1
% Ground/Roof const.: (x1G + x2nG + x2FG +x2PG)./cost <= Area_G %(nsub*3)
%                     (x1R + x2nR + x2FR +x2PR).,cost <= Area_R

ta = repmat(TA,[3,1]);
InvToImp = ta(:)./EcostALL';
for j = 1:nsub
    for i = 1:m1
% ground area
        InvToGround = TA(1:3)./EcostALL((j-1)+(1:nsub:nsub*3));
        A(nconst+(j-1)*m1*2+(i-1)*2+1,(j-1)+(1:nsub:nsub*3))=  InvToGround;  % x1 (m1)
        A(nconst+(j-1)*m1*2+(i-1)*2+1,n1*i+(j-1)+(1:nsub:nsub*3))= InvToGround;   % x2n
        A(nconst+(j-1)*m1*2+(i-1)*2+1,n1*m1+n1*i+(j-1)+(1:nsub:nsub*3))= InvToGround;    % x2F
        A(nconst+(j-1)*m1*2+(i-1)*2+1,n1*m1*2+n1*i+(j-1)+(1:nsub:nsub*3))= InvToGround;    % x2P   
% roof area
        InvToRoof = TA(4:5)./EcostALL((j-1)+(10:nsub:nsub*5));
        A(nconst+(j-1)*m1*2+ i*2,(j-1)+(10:nsub:nsub*5))= InvToRoof;   % x1 (m1)
        A(nconst+(j-1)*m1*2+ i*2,n1*i+(j-1)+(10:nsub:nsub*5))= InvToRoof;    % x2n
        A(nconst+(j-1)*m1*2+ i*2,n1*m1+n1*i+(j-1)+(10:nsub:nsub*5))= InvToRoof;    % x2F
        A(nconst+(j-1)*m1*2+ i*2,n1*m1*2+n1*i+(j-1)+(10:nsub:nsub*5))= InvToRoof;    % x2P   
    end
end
nconst=nconst+2*nsub*m1;
    
% linearization constraint
% Learning var: L:(LN,LF,LP)
% x2n:  x2n(i)-M*(LN)<=0 for all s  
    A(nconst+(1:n1*m1),n1+(1:n1*m1))=eye(n1*m1);               % x2
    A(nconst+(1:n1*m1),+n1+n1*m1*3+(1:n1))=-repmat(M*eye(n1),[m1,1]); % M*LN
    nconst=nconst+n1*m1;
% x2f: x2f-M*LF<= 0 for all s in S
    A(nconst+(1:n1*m1),n1+n1*m1+(1:n1*m1))=eye(n1*m1); % x2_FL
    A(nconst+(1:n1*m1),n1+n1*m1*3+n1+(1:n1))=-M*repmat(eye(n1),[m1,1]); % M*LF                      % -M*LF(i)
    nconst=nconst+n1*m1;
% x2p: x2p-M*LP<= 0 for all s in S
    A(nconst+(1:n1*m1),n1+n1*m1*2+(1:n1*m1))=eye(n1*m1); % x2_LP
    A(nconst+(1:n1*m1),n1+n1*m1*3+n1*2+(1:n1))=-M*repmat(eye(n1),[m1,1]); % M*LP                      % -M*LF(i)
    nconst=nconst+n1*m1; 
% l1:learning const. 
% a. -x1(u)-Th((PL)*LF(v)-Th(PL)*LF(w)+Th(PL)*LP(u) <= 0 for i in n1
%  # n1
    A(nconst+(1:n1),(1:n1))=-eye(n1);       % x1
    A(nconst+(1:n1),(1:n1)+n1+n1*m1*3+n1*2)=diag(LT1);  % Th(PL)*LP(u)
    block = ones(nsub)-diag(ones(1,nsub));
    for i=1:nGI
        A(nconst+(i-1)*nsub+(1:nsub),n1+n1*m1*3+n1+(i-1)*nsub+(1:nsub))= -block.*LT1((i-1)*nsub+(1:nsub))';  % -Th(PL)*(LF(v)+LF(w))
    end
    nconst=nconst+n1;          
% b. x1-M(LF)<= Th(FL) for i in n1                                # = n1
    A(nconst+(1:n1),(1:n1))=eye(n1);                            % x1
%      A(nconst+(1:n1),(1:n1)+n1+n1*m1*3)=-M*eye(n1);           % LN
    A(nconst+(1:n1),(1:n1)+n1+n1*m1*3+n1)=-M*eye(n1);           % LF
     nconst=nconst+n1;     
% c. -x1+Th(FL)*LF<=0                                             # = n1
    A(nconst+(1:n1),(1:n1))=-eye(n1);                           % x1     
    A(nconst+(1:n1),(1:n1)+n1+n1*m1*3+n1)=diag(LT2);            % LF
     nconst=nconst+n1;
% d. x1-M(LF+LP)<= TH(PL)                                         # = n1
    A(nconst+(1:n1),(1:n1))=eye(n1);                            % x1     
    A(nconst+(1:n1),(1:n1)+n1+n1*m1*3+n1)=-M*eye(n1);           % LF 
    A(nconst+(1:n1),(1:n1)+n1+n1*m1*3+n1*2)=-M*eye(n1);         % LP    
     nconst=nconst+n1;
% e. LN+LF+LP=1  for all i in I ** equality constraints
    Aeq=zeros(n1,ndv);
    Aeq(1:n1,(1:n1)+n1+n1*m1*3)=eye(n1);                        % LN 
    Aeq(1:n1,(1:n1)+n1+n1*m1*3+n1)=eye(n1);                     % LF 
    Aeq(1:n1,(1:n1)+n1+n1*m1*3+n1*2)=eye(n1);                   % LP 
    beq=[ones(n1,1)]  ;   
% h. 2*LF(u)-LF(v)-LF(w)-LP(v)-LP(w)<=0                           # = n1
    block2 =  2*eye(nsub)-block;
    for i=1:nGI
        A(nconst+(i-1)*nsub+(1:nsub),n1+n1*m1*3+n1+(i-1)*nsub+(1:nsub)) = block2;         % LF(u)
        A(nconst+(i-1)*nsub+(1:nsub),n1+n1*m1*3+n1*2+(i-1)*nsub+(1:nsub))=-block;  % -(LP(v)+LP(w))
    end    
    nconst=nconst+n1;
% f: coefficients of [x1(n1),x2(n1*m1),x2_FL(n1*m1),x2_PL(n1*m1),L(n1*r1),VaR(1),z(m1*m2)]
%CVaR constraints
% (1) rho-fj-zj<=0 (m1*m2) % important: what's the coeff. of CVaR constraints!!!

    for j=1:m1                                                 
        A(nconst+m2*(j-1)+(1:m2),(1:n1))= -repmat(s12(:,j)',[m2,1]);      % -s1*x1*deterioration d1*d2                                                                                           
        for k=1:m2                                  
            A(nconst+m2*(j-1)+k,(1:n1)+n1+n1*(j-1))= -s2n(:,k+(j-1)*m2)';          % -s2n*x2_NL
            A(nconst+m2*(j-1)+k,(1:n1)+n1+n1*m1+n1*(j-1))= -s2f(:,k+(j-1)*m2)';    % -s2f*x2_FL    
            A(nconst+m2*(j-1)+k,(1:n1)+n1+n1*m1*2+n1*(j-1))=...                    % -s2p*x2_PL
                -s2p(:,k+(j-1)*m2)'; 
        end      
    end
    A(nconst+(1:m1*m2),n1+n1*m1*3+n1*r1+1)=ones(m1*m2,1); % rho
    A(nconst+(1:m1*m2),n1+n1*m1*3+n1*r1+1+(1:m1*m2))=-eye(m1*m2); % zj      
    nconst=nconst+m1*m2;
% (2) rho - 1/(1-alpha)*sum(Pjk*zjk)>=CVAR ->
%     -rho+1/(1-alpha)*sum(Pjk*zjk)<=-CVAR
    A(nconst+1,n1+n1*m1*3+n1*r1+1+(1:m1*m2))=...       %[zjk]
        p1*p2*ones(1,m1*m2)/(alpha);
    A(nconst+1,n1+n1*m1*3+n1*r1+1)=-1;        %-rho
    nconst = nconst+1;

b=[repmat(Budget,[m1,1]);repmat(ub',[m1,1]);G_R_A(:);zeros(n1*m1*3+n1,1);...
    (LT2')-tol;zeros(n1,1);(LT1'-tol);zeros(n1,1);zeros(m1*m2,1);-CVAR]; 
% budget(m1*m2),UB(n1*m1*m2),linearization(n1+n1*m1+n1*m1*m2),
%        learning(n1*r1),zjk(m1*m2),CVaR(1)
intcon=n1+n1*m1*3+(1:n1*r1); % indicators of binary var.  
LB=[zeros(1,n1+n1*m1*3+n1*r1),-inf,zeros(1,m1*m2)];  %lower bound
UB=[ub,repmat(ub,[1,m1*3]),ones(1,n1*r1),inf*ones(1,1+m1*m2)];    %upper bound- all variables are positive

%% Solve model
[x, fval, exitflag] = intlinprog(f,intcon,A,b,Aeq,beq,LB,UB);
-fval; %(M-m3)
l1=reshape(x(n1+n1*m1*3+(1:n1*r1)),[n1,3]);  % learning var [row col]
x(n1+n1*m1+n1*m1+n1*m1+n1*r1+1+(1:m1*m2)); %z 
x1=reshape(x(1:n1),[n1,1]);                % stage I decisions
x2n=reshape(x(n1+(1:n1*m1)),[n1,m1]);      % stage II decisions- no learning
x2f=reshape(x(n1+n1*m1+(1:n1*m1)),[n1,m1]); % stage II - full learning
x2p=reshape(x(n1+2*n1*m1+(1:n1*m1)),[n1,m1]); % stage II - partial learning
y1=-A(m1+n1*m1+2*nsub*m1+(n1*m1*r1)+n1*5+(1:m1*m2),...
    (1:(n1+n1*m1*3)))*x(1:(n1+n1*m1*3));       
yy=sort(y1);
TrueCVAR=mean(yy(1:ceil(m1*m2*(alpha+eps))))
result=A*x;

%% loops - CVaR
bt=1;iterb=1;iterk=40;
maxCVAR=3.0*10^6;
bb= 1.8*10^8;  % 180 M Budget
kk = 2.2*10^6+(1:iterk)*(maxCVAR-2.2*10^6)/iterk;  %CVAR target
[bi,bj]=size(b);
[ci,~]=size(kk');
xx=zeros(ndv+1+1,ci); 
CVAR3d=zeros(iterb,ci);    %CVAR
VAR3d=zeros(iterb,ci);     %VAR
Bi3d=zeros(iterb,ci);    %total benefit
MEAN3d=zeros(iterb,ci);    % mean benefit at 2nd stage

    for i=1:(ci)
%         b(bi)=-kk(i);
        b=[repmat(bb,[m1,1]);repmat(ub',[m1,1]);G_R_A(:);zeros(n1*m1*3+n1,1);...
        (LT2')-tol;zeros(n1,1);(LT1'-tol);zeros(n1,1);zeros(m1*m2,1);-kk(i)];
        disp(-b(bi))
       [x, fval, exitflag] = intlinprog(f,intcon,A,b,Aeq,beq,LB,UB);
         if exitflag<=0
            x=zeros(ndv,1);fval=0;
            break;
         else          
            y1=-A(m1+n1*m1+2*nsub*m1+(n1*m1*r1)+n1*5+(1:m1*m2),...
                (1:(n1+n1*m1*3)))*x(1:(n1+n1*m1*3));  
            
            yy=sort(y1);
            TrueCVAR=mean(yy(1:ceil(m1*m2*(alpha+eps))))
            TrueVAR=yy(ceil(m1*m2*(alpha)));
            CVAR3d(bt,i)=TrueCVAR;         
            VAR3d(bt,i)=TrueVAR;
            Bi3d(bt,i)=-fval;                
            MEAN3d(bt,i)=mean(yy);
         end
        % Calculate true CVAR
        xx(:,i)=[x;-fval;exitflag]';  % [ndv-solution; Budget; exitflag]
%        nviolation=sum(x(n+1+(1:m))>1.0e-8)    
    end
    
%% Solution plots
% save ('TI_CRN_0720','xx','CVAR3d','VAR3d','MEAN3d','f') 
sim = 6;
% cvar_v = kk(sim);  
l1 = reshape(xx(n1+n1*m1*3+(1:n1*r1),sim),[n1,3]) ;  % learning var [row col]
x1 = reshape(xx(1:n1,sim),[n1,1]);                   % stage I decisions
x2n = reshape(xx(n1+(1:n1*m1),sim),[n1,m1]);         % stage II decisions- no learning
x2f = reshape(xx(n1+n1*m1+(1:n1*m1),sim),[n1,m1]);   % stage II - advance learning
x2p = reshape(xx(n1+2*n1*m1+(1:n1*m1),sim),[n1,m1]); % stage II - basic learning
% b=[repmat(Budget,[m1,1]);repmat(ub',[m1,1]);zeros(n1*m1*3+n1,1);...
%     (LT2');zeros(n1,1);(LT1');zeros(n1,1); zeros(m1*m2,1);-CVAR]; 
b=[repmat(Budget,[m1,1]);repmat(ub',[m1,1]);G_R_A(:);zeros(n1*m1*3+n1,1);...
    (LT2');zeros(n1,1);(LT1');zeros(n1,1);zeros(m1*m2,1);-CVAR]; 
y1=-A(m1+n1*m1+2*nsub*m1+(n1*m1*3)+n1*5+(1:m1*m2),...
               (1:(n1+n1*m1+n1*m1+n1*m1)))*x(1:(n1+n1*m1+n1*m1+n1*m1));        
yy=sort(y1);
mean(yy(1:ceil(m1*m2*(alpha))))

[x1,LT1',LT2',ub']
%% Plot the Stromwater Reduction Distribution of the first and last Solutions
results = A* xx(1:ndv,1);
rho_s = xx(n1+n1*m1+n1*m1+n1*m1+n1*r1+1,[1,27]);
f_nst = m1+n1*m1+m1*2*nsub+(n1*m1*3)+n1*5+(1:m1*m2); 
f_values_1 = -A(m1+n1*m1+2*nsub*m1+(n1*m1*3)+n1*5+(1:m1*m2),...
               (1:(n1+n1*m1+n1*m1+n1*m1)))*xx(1:(n1+n1*m1+n1*m1+n1*m1),1);   % the stormwtaer reduction of all scenarios (m1*m2) of simulation 1
f_values_27 =  -A(m1+n1*m1+2*nsub*m1+(n1*m1*3)+n1*5+(1:m1*m2),...
               (1:(n1+n1*m1+n1*m1+n1*m1)))*xx(1:(n1+n1*m1+n1*m1+n1*m1),27); % the stormwtaer reduction of all scenarios (m1*m2) of simulation 27
bins = (2e6:1e5:4e6)
h1 = histogram(f_values_1)  % the histogram of the first solution
hold on;
h2 = histogram(f_values_27)  % the histogram of the last solution
h1.Normalization = 'probability';
h1.BinWidth = 1e5;
h2.Normalization = 'probability';
h2.BinWidth = 1e5;
hold off



%% results organization
bt=1;
[ci,cj]=size(kk');
ii=find(CVAR3d(bt,:)>0);
xx1=reshape(xx(1:n1,:),[n1,iterk]);  % first stage investment
xx2 = zeros(nGI, iterk);
for j = 1:iterk
    for t = 1:nGI
        xx2(t,j) = sum(xx1((t-1)*nsub+(1:nsub),j));
    end
end
    
xx3=zeros(3,ci);
% xx5(1,:)=xx(ndv+1,bt,:);
xx3(1,:)=xx(ndv+1,:); % the objective value
Obj=xx3(1,ii);
% ii=find(xx5(1,:)>0);
xx3(2,:)=MEAN3d(bt,:); % Stage II average
xx3(3,:)=CVAR3d(bt,:); % Stage II average  
%% PLOT
fig1=figure(1); % fval vs CVaR

plot(CVAR3d(bt,ii),xx3(1,ii),'r-o',CVAR3d(bt,ii),xx3(2,ii),'b-*')
C={'Program Average','Stage II Average'};
legend('Location','northwest');
legend(C);
grid on;xlabel('CVaR Value (m^3/yr)');ylabel('Expected Runoff Reduction (m^3)');
%title('Benefit Received: Total Cumulative VS Stage II only');
set(fig1,'position',[0 0 300 300]);
set(gcf,'PaperPositionMode','auto')

colors =['b','m','g','c','r','k'];
markers = ['b-o','m-*','g-.'];
%%
fig2=figure(2); % increase CVaR target - the dicision mixed
left_color = [0.1 0.1 0.1 ];
right_color = [1 0 0];
set(fig2,'defaultAxesColorOrder',[left_color; right_color]);
yyaxis left
hplot1=plot(CVAR3d(bt,ii),xx2(1,ii),'m-o',CVAR3d(bt,ii),xx2(2,ii),'b-*',CVAR3d(bt,ii),xx2(3,ii),'g->'...
    ,CVAR3d(bt,ii),xx2(4,ii),'c-<',CVAR3d(bt,ii),xx2(5,ii),'k-+');
ylim([0 6E6]);
xlabel('CVaR(m^3/yr)');ylabel('Investment($/yr)');
yyaxis right
hplot2=plot(CVAR3d(bt,ii),xx3(1,ii),'--','Linewidth',2 );
ylabel('Expected Runoff Reduction (m^3/yr)');grid on;
ylim([2.5E6 3.2E6]);

C={'RG','IT','PP','RB','GR','Runoff Reduction'};
legend('Location','northwest');
hlegend1 = legend(C);       

set(fig2,'position',[700 350 700 300]);
set(gcf,'PaperPositionMode','auto');
% set(gcf, 'PaperSize', [6 3]);
%% Plot Subsewer
% xx1=reshape(xx(1:n1,:),[n1,iterk]);
xx4 = zeros(nsub, iterk);
for j = 1:iterk
    for t = 1:nsub
        xx4(t,j) = sum(xx1(t:3:15,j));
    end
end

fig3=figure(3); % fval vs CVaR

plot(CVAR3d(bt,ii),xx4(1,ii),'r-o',CVAR3d(bt,ii),xx4(2,ii),'b-*',CVAR3d(bt,ii),xx4(3,ii),'g-*')
C={'Upper','Middle', 'Lower'};
legend('Location','northeast');
legend(C);
grid on;xlabel('CVaR Value (m^3/yr)');ylabel('Investment($/yr)');
%title('Benefit Received: Total Cumulative VS Stage II only');
set(fig3,'position',[0 350 300 300]);
set(gcf,'PaperPositionMode','auto')

%% Stage I vs Stage II
x2npf = sum(xx(n1+(1:n1*m1*3),:))/m1;
xx5 = zeros(2, iterk);
xx5(1,:) = sum(xx1);
xx5(2,:) = x2npf;
fig4=figure(4); % fval vs CVaR

plot(CVAR3d(bt,ii),xx5(1,ii),'r-o',CVAR3d(bt,ii),xx5(2,ii),'b-o')
C={'Stage I','Stage II'};
legend('Location','northeast');
legend(C);
grid on;xlabel('CVaR Value (m^3/yr)');ylabel('Investment($/yr)');
%title('Benefit Received: Total Cumulative VS Stage II only');
set(fig4,'position',[700 0 300 300]);
set(gcf,'PaperPositionMode','auto')
%% Treated Area
TAa =repmat(TA,[3,1]);
costTA = TAa(:)./EcostALL';  % cost per treated area
xx6 = xx1.*costTA ;

Ground_A'./costTA(4:6)  % the cost of using IT to treat all impervious
Roof_A'./costTA(10:12)  % the cost of using RB to treat all roof
MinCost = sum(Ground_A'./costTA(4:6)+ Roof_A'./costTA(10:12)) *25

xx7G = zeros(3, iterk);
xx7G(1,:) = sum(xx6(1:3:9,:));  % Upper's ground area treated 
xx7G(2,:) = sum(xx6(2:3:9,:));  % Middle's ground area treated 
xx7G(3,:) = sum(xx6(3:3:9,:));  % Lower's ground area treated 
xx7GA = sum(xx7G)
xx7R = zeros(3, iterk);
xx7R(1,:) = sum(xx6(9+(1:3:6),:));  % Upper's ground area treated 
xx7R(2,:) = sum(xx6(9+(2:3:6),:));  % Middle's ground area treated 
xx7R(3,:) = sum(xx6(9+(3:3:6),:));  % Lower's ground area treated 
xx7RA = sum(xx7R)

fig5=figure(5); % fval vs CVaR

plot(CVAR3d(bt,ii),xx7GA(ii),'r-o',CVAR3d(bt,ii),xx7RA(ii),'b-o')
C={'Ground Treated','Roof Treated'};
legend('Location','northeast');
legend(C);
grid on;xlabel('CVaR Value (m^3/yr)');ylabel('Investment($/yr)');
%title('Benefit Received: Total Cumulative VS Stage II only');
set(fig5,'position',[700 0 300 300]);
set(gcf,'PaperPositionMode','auto')

%% print figures
print(fig1,'C:\School\3rd paper - case study\02Optimization\Fig20191031_test\fig1_transLearning_obj.tif','-dtiff','-r500')  % save figures
print(fig2,'C:\School\3rd paper - case study\02Optimization\Fig20191031_test\fig2_transLearning_SMP.tif','-dtiff','-r500')  % save figures
print(fig3,'C:\School\3rd paper - case study\02Optimization\Fig20191031_test\fig3_transLearning_sub.tif','-dtiff','-r500')  % save figures
print(fig4,'C:\School\3rd paper - case study\02Optimization\Fig20191031_test\fig4_transLearning_now_later.tif','-dtiff','-r500')  % save figures
print(fig5,'C:\School\3rd paper - case study\02Optimization\Fig20191031_test\fig5_transLearning_treated_area.tif','-dtiff','-r500')  % save figures
save ('C:\School\3rd paper - case study\02Optimization\Fig20191031_test\TI_CRN_CVAR_trans_learning1204','xx','CVAR3d','VAR3d','MEAN3d','f', 'A','kk') 

%% Save to Excel file
filename ='C:\School\3rd paper - case study\02Optimization\Fig20191031_test\solutions_and_plot.xlsx';
AA = {xx1};
sheet = 'Stage I Solutions';
xlRange = 'A1';
xlswrite(filename,xx1,'Stage I Solutions',xlRange)
xlswrite(filename,xx3,'Obj_CVAR',xlRange)

%% No Learning  - The same samples
%% Solve model
[x, fval, exitflag] = intlinprog(f,intcon,A,b,Aeq,beq,LB,UB);
-fval/10^6; %(M-m3)
l1=reshape(x(n1+n1*m1*3+(1:n1*r1)),[n1,3]);  % learning var [row col]
x(n1+n1*m1+n1*m1+n1*m1+n1*r1+1+(1:m1*m2)); %z 
x1=reshape(x(1:n1),[n1,1]);                % stage I decisions
x2n=reshape(x(n1+(1:n1*m1)),[n1,m1]);      % stage II decisions- no learning
x2f=reshape(x(n1+n1*m1+(1:n1*m1)),[n1,m1]); % stage II - full learning
x2p=reshape(x(n1+2*n1*m1+(1:n1*m1)),[n1,m1]); % stage II - partial learning
y1=-A(m1+n1*m1+2*nsub*m1+(n1*m1*r1)+n1*5+(1:m1*m2),...
    (1:(n1+n1*m1*3)))*x(1:(n1+n1*m1*3));       
yy=sort(y1);
mean(yy(1:ceil(m1*m2*(alpha+eps))))
result=A*x;



