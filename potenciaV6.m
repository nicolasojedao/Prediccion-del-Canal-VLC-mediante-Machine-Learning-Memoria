
clear; clc;
rng('shuffle'); 


lx = 20; 
ly = 5;  
lz = 5;  

Pt = 10; 


% Multiplicador 10 genera exactamente 10251 filas
Nx = lx * 10; 
Ny = ly * 10; 
Nz = lz * 10; 

x_i = 3;   y_i = 0.5; z_i = 4.5;
alpha_i = 45; beta_i = 45;

Aw = 1; pw = 0.6; 
ang_rad = 60; 
m = -log(2)/log(abs(cos(ang_rad*pi/180))); 
Ap = 0.0001; 
eta = 1.5; 
fov = 70; 



% muro invisible
gv = [2000 6000]; fv = [0 5];   

W = 6; H = 5; X = 6; Y = 3; 
es = 5; t = 5*10^-9;        
gymma = 0.017; g = 0.72; f = 0.5; 
kr = [0.1 0.01]; km = [0 10]; ks = kr+km; 
N = 40; p = 0.1; c = 3*10^8; 



%% 2 INICIALIZACION DEL ENTORNO ESPACIAL
dA = lz*ly/(Ny*Nz); 
x = linspace(0.5, lx-0.5, Nx+1); 
y = linspace(0.5, ly-0.5, Ny+1);  
z = 0:lz/Nz:lz;  

% OPTIMIZACION DE MEMORIA
total_iteraciones = (Nx+1) * (Ny+1);
dataset_VLC = zeros(total_iteraciones, 10); 
fila_actual = 1;



%% 3 EXTRACCION DE DATOS 
disp(['iniciando simulacion  (', num2str(total_iteraciones), ' filas).']);


Nx_wall = lx * 1; 
Nz_wall = lz * 1; 

x_wall = linspace(0.5, lx-0.5, Nx_wall+1); 
z_wall = 0:lz/Nz_wall:lz; 

for ii = 1:Nx+1
    for jj = 1:Ny+1
        
        alpha_j = 180 * rand(); 
        beta_j = 90 * rand();   
        
        % Z VARIABLE 
        z_rx = 1.5 + (1.9 - 1.5) * rand(); 
        
        % Z CONSTANTE 
        % z_rx = 1.8;
        



        % linea de Vista (LOS)
        incidencia_rad = incline(x_i, y_i, z_i, x(ii), y(jj), z_rx, alpha_j, beta_j);
        incidencia_deg = rad2deg(incidencia_rad); 
        irradiancia_rad = rotacion(x(ii), y(jj), z_rx, x_i, y_i, z_i, alpha_i, beta_i);
        
        [m_HLoS1, ~] = HLoS_direct(x_i,y_i,z_i, x(ii),y(jj),z_rx, Ap,eta,alpha_i,alpha_j,beta_i,beta_j, incidencia_rad,incidencia_deg, m,fov,gv,fv,W,H,X,Y,t,es,c);
        
        % reflexiones (NLOS) 
        re = zeros(Nx_wall+1, Nz_wall+1);
        for kk = 1:Nx_wall+1  
            for ll = 1:Nz_wall+1
                r_rand = randi([0 90],1,1); 
                s_rand = randi([0 90],1,1); 
                incidenciaw_pru = incline(x_wall(kk), 0.01, z_wall(ll), x(ii), y(jj), z_rx, alpha_j, beta_j); 
                incidenciawpru = rad2deg(incidenciaw_pru);
                [m_HnLoSp, ~] = HnLos_calculation(x_i,y_i,z_i, x(ii),y(jj),z_rx, x_wall(kk),0.01,z_wall(ll), dA/70,pw,alpha_i,alpha_j,r_rand,beta_i,beta_j,s_rand,Ap,incidenciawpru,incidenciaw_pru,eta,m,fov,gv,fv,W,H,X,Y,t,es,c);
                re(kk,ll) = m_HnLoSp;
            end 
        end
        ret = sum(reshape(re,1,[]));
        
        re1 = zeros(Nx_wall+1, Nz_wall+1);
        for kk = 1:Nx_wall+1  
            for ll = 1:Nz_wall+1
                r1_rand = randi([0 90],1,1); 
                s1_rand = randi([0 90],1,1); 
                incidenciaw_pru1 = incline(x_wall(kk), ly-0.01, z_wall(ll), x(ii), y(jj), z_rx, alpha_j, beta_j); 
                incidenciawpru1 = rad2deg(incidenciaw_pru1);
                [m_HnLoSp1, ~] = HnLos_calculation(x_i,y_i,z_i, x(ii),y(jj),z_rx, x_wall(kk),ly-0.01,z_wall(ll), dA/70,pw,alpha_i,alpha_j,r1_rand,beta_i,beta_j,s1_rand,Ap,incidenciawpru1,incidenciaw_pru1,eta,m,fov,gv,fv,W,H,X,Y,t,es,c);
                re1(kk,ll) = m_HnLoSp1;
            end 
        end
        ret1 = sum(reshape(re1,1,[]));
        H_NLoS_Total = ret + ret1;
        

        
        % polvo udsm
        [m_Hscat1, ~] = H_scater(x_i,y_i,z_i, x(ii),y(jj),z_rx, Ap,m,f,g,gymma,kr,km,ks,p,N, rad2deg(irradiancia_rad),c,alpha_i,beta_i);
        Hsca = sum(m_Hscat1);
        


        % suma total y potencia
        H0_total = m_HLoS1 + H_NLoS_Total + Hsca;
        P_rx = H0_total * Pt;
        


        % dataset completo
        dataset_VLC(fila_actual, :) = [x(ii), y(jj), z_rx, alpha_j, beta_j, m_HLoS1, H_NLoS_Total, Hsca, H0_total, P_rx];
        fila_actual = fila_actual + 1;
        
    end 
end 
writematrix(dataset_VLC, 'dataset_vlc_z_variable_10k_FINAL.csv');
disp('simulacion finalizada');


% FUNCIONES AUXILIARES - NO TOCARLAS

function ang_inc=incline(x1,y1,z1,x2,y2,z2,alpha,beta)
    v=[x1-x2,y1-y2,z1-z2];
    Ntilt=[cosd(alpha)*sind(beta),sind(alpha)*sind(beta),cosd(beta)];
    d_p=dot_product(v,Ntilt);
    d=sqrt((x1-x2)^2+(y1-y2)^2+(z1-z2)^2);
    ang_inc=acos(d_p/d);
end
function ang_incidencia=rotacion(x1,y1,z1,x2,y2,z2,alpha,beta)
    v=[x1-x2,y1-y2,z1-z2];
    Ntilt=[cosd(alpha)*sind(beta),sind(alpha)*sind(beta),-cosd(beta)];
    d_p=dot_product(v,Ntilt);
    d=sqrt((x1-x2)^2+(y1-y2)^2+(z1-z2)^2);
    ang_incidencia=acos(d_p/d);
end
function[m_HnLoS,dm]=HnLos_calculation(x_i,y_i,z_i,x_j,y_j,z_j,x_w,y_w,z_w,Aw,pw,alpha_i,alpha_j,alpha_w,beta_i,beta_j,beta_w,Ap,inc,inc_r,eta,m,fov,gv,fv,W,H,X,Y,t,es,c)
    dv_iw= dv(x_i,y_i,x_w,y_w,fv);
    sv_iw= sv(x_i,y_i,z_i,x_w,y_w,z_w,fv);
    Piw=P_expt(gv,fv,W,H,X,Y,t,es, dv_iw,sv_iw);
    dv_wj= dv(x_w,y_w,x_j,y_j,fv);
    sv_wj= sv(x_w,y_w,z_w,x_j,y_j,z_j,fv);
    Pwj=P_expt(gv,fv,W,H,X,Y,t,es, dv_wj,sv_wj);
    g=gain(eta,inc,inc_r,fov);
    [v1,d1]=point_to_vector(x_i,y_i,z_i,x_w,y_w,z_w);
    Nnorm1=norm_vec_trans(alpha_i,beta_i);
    p1=dot_product(v1,Nnorm1);
    [v2,d2]=point_to_vector(x_w,y_w,z_w,x_i,y_i,z_i);
    Nnorm2=norm_vec_receiver(alpha_w,beta_w);
    p2=dot_product(v2,Nnorm2);
    [v3,d3]=point_to_vector(x_w,y_w,z_w,x_j,y_j,z_j);
    Nnorm3=norm_vec_receiver(alpha_w,beta_w);
    p3=dot_product(v3,Nnorm3);
    [v4,d4]=point_to_vector(x_j,y_j,z_j,x_w,y_w,z_w);
    Nnorm4=norm_vec_receiver(alpha_j,beta_j);
    p4=dot_product(v4,Nnorm4);
    digits(2);
    dm=((d1+d3)/c);
    dm=vpa(dm);
    dm=double(subs(dm));
    m_HnLoS= abs(((m+1)*Ap*Aw*pw*p1*p2*p3*p4*g*Piw*Pwj)/((d1^2)*(d3^2)*d1*d2*d3*d4));
    m_HnLoS=vpa(m_HnLoS);
    m_HnLoS=double(subs(m_HnLoS));
end
function [m_HLoS,dm]=HLoS_direct(x_i,y_i,z_i,x_j,y_j,z_j,Ap,eta,alpha_i,alpha_j,beta_i,beta_j,incidencia,incidencia_r,m,fov,gv,fv,W,H,X,Y,t,es,c)
    dv_ij= dv(x_i,y_i,x_j,y_j,fv);
    sv_ij= sv(x_i,y_i,z_i,x_j,y_j,z_j,fv);
    Pij=P_expt(gv,fv,W,H,X,Y,t,es, dv_ij,sv_ij);
    [v1,d1]=point_to_vector(x_i,y_i,z_i,x_j,y_j,z_j);
    Nnorm1=norm_vec_trans(alpha_i,beta_i);
    p1=dot_product(v1,Nnorm1);
    [v2,d2]=point_to_vector(x_j,y_j,z_j,x_i,y_i,z_i);
    Nnorm2=norm_vec_receiver(alpha_j,beta_j);
    p2=dot_product(v2,Nnorm2);
    g=gain(eta,incidencia,incidencia_r,fov);
    digits(2);
    dm= d1/c;
    dm=vpa(dm);
    dm=double(subs(dm));
    if (incidencia>=0) && (incidencia<=fov)
        m_HLoS=abs(((m+1)*Ap/(2*3.1416*d1^2))*(p1^m/d1)*(p2/d2)*g* Pij);
        m_HLoS=vpa(m_HLoS);
        m_HLoS=double(subs(m_HLoS));
    else
        m_HLoS=0;
    end
end
function [m_total_Hscat,dm_total]=H_scater(x_i,y_i,z_i,x_j,y_j,z_j,Ap,m,f,g,gymma,kr,km,ks,p,N,theta_ij,c,alpha_i,beta_i)
    [v1,dij]=point_to_vector(x_i,y_i,z_i,x_j,y_j,z_j);
    dm_total=zeros(1,N);
    m_total_Hscat=zeros(1,N);
    for i =1:N
        Rr=0.5;
        rn=Rr*rand(1,1);
        theta_sn_j=randi([-180 180]);
        B_i_sn=Bisn(theta_sn_j,theta_ij);
        xs=rn*cosd(B_i_sn);
        ys=rn*sind(B_i_sn);
        zs=rn*cosd(theta_sn_j);
        phi_i_sn_radian=phi_scater(xs,ys,zs,x_i,y_i,z_i,alpha_i,beta_i);
        phi_i_sn=rad2deg(phi_i_sn_radian);
        di_sn=sqrt(rn^2+dij^2-2*rn*dij*cosd(B_i_sn));
        Di_j=di_sn+rn;
        Gn=Gain_n(f,g,gymma,phi_i_sn,kr,km,ks,p,N);
        digits(2);
        dm=(Di_j/c);
        dm=vpa(dm);
        dm=double(subs(dm));
        dm_total(i)=+dm;
        if (theta_sn_j>=-180) && (theta_sn_j<=180)
            Hscat=abs(((m+1)*Ap*Gn/(2*3.1416*Di_j^2))*(cosd(phi_i_sn))^m*cosd(theta_sn_j));
            Hscat=vpa(Hscat);
            Hscat=double(subs(Hscat));
        else
            Hscat=0;
        end
        m_total_Hscat(i)=+Hscat;
    end
end
function Pij=P_expt(gv,fv,W,H,X,Y,t,es,d_v,s_v)
    syms w h x y p E
    if(gv(1)>=2*d_v) &&(gv(2)>=s_v)
        w_int=int(gv(1),w,0,W);
        h_int=int(gv(2),h,0,H);
        A=[w_int h_int];
        x_int=int(fv(1),x,0,X);
        y_int=int(fv(2),y,0,Y);
        B=[x_int y_int];
        exp_value=dot(A,B);
        f=p*t;
        est=-es*exp_value;
        d=vpa(subs(f,p,est),8);
        f=exp(E);
        Pij=vpa(subs(f,E,d),4);
    else
        Pij=0;
    end
end
function d_v = dv(x1, y1, x2, y2, fv)
    denominador = sqrt((y1-y2)^2 + (x1-x2)^2);
    if denominador == 0
        d_v = 0; 
    else
        d_v = abs((y1-y2)*fv(1) - (x1-x2)*fv(2) - x2*y1 + x1*y2) / denominador;
    end
end
function s_v= sv(x1,y1,z1,x2,y2,z2,fv)
    if(z1<=z2)
        s_v=((y1-y2)^2+(x1-x2)^2+(fv(1)-x1)^2+(fv(2)-y1)^2-[(fv(1)-x2)^2+(fv(2)-y2)^2]/2*sqrt((y1-y2)^2+(x1-x2)^2))+z1;
    else
        s_v=((y1-y2)^2+(x2-x1)^2+(fv(1)-x2)^2+(fv(2)-y2)^2-[(fv(1)-x1)^2+(fv(2)-y1)^2]/2*sqrt((y1-y2)^2+(x2-x1)^2))+z2;
    end
end
function g=gain(eta,incide,incide_r,fov)
    if  (incide_r<=fov) && (incide_r>=0)
        g=(eta^2)/(sind(fov)^2);
    else
        g=0;
    end
end
function f=dot_product(a,b)
    f=dot(a,b);
end
function n=norm_vec_trans(alpha,beta)
    n=[cosd(alpha)*sind(beta),sind(alpha)*sind(beta),-cosd(beta)];
end
function m=norm_vec_receiver(alpha,beta)
    m=[cosd(alpha)*sind(beta),sind(alpha)*sind(beta),cosd(beta)];
end
function [Vec,lenght]= point_to_vector(x1,y1,z1,x2,y2,z2)
    Vec=[x2-x1,y2-y1,z2-z1];
    lenght=sqrt((x2-x1)^2+(y2-y1)^2+(z2-z1)^2);
end
function Gn=Gain_n(f,g,gymma,phi_i_sn,kr,km,ks,p,N)
    p_mie=pmie(f,g,phi_i_sn);
    p_ray=pray(gymma,phi_i_sn);
    p_total=(kr/ks)*p_ray+(km/ks)*p_mie;
    f_scat=p_total*sind(phi_i_sn);
    Gn=p*f_scat/N;
end
function p_mie=pmie(f,g,phi_i_sn)
    p_mie=(1-g^2/4*pi)*(1/(1+g^2-2*g*cosd(phi_i_sn))^1.5+ f*3*(cosd(phi_i_sn))^2-1/2*(1+g^2)^1.5);
end
function p_ray=pray(gymma,phi_i_sn)
    p_ray=3*[1+3*gymma+(1-gymma)*(cosd(phi_i_sn))^2]/(16*pi*(1+2*gymma));
end
function phi_i_sn=phi_scater(x1,y1,z1,x2,y2,z2,alpha,beta)
    v=[x1-x2,y1-y2,z1-z2];
    Ntilt=[cosd(alpha)*sind(beta),sind(alpha)*sind(beta),-cosd(beta)];
    d_p=dot_product(v,Ntilt);
    d=sqrt((x1-x2)^2+(y1-y2)^2+(z1-z2)^2);
    phi_i_sn=acos(d_p/d);
end
function B_i_sn=Bisn(theta_sn_j,theta_ij)
    if(theta_ij<theta_sn_j)
        B_i_sn=theta_sn_j-theta_ij;
    elseif(theta_sn_j<=theta_ij)
        B_i_sn= theta_ij-theta_sn_j;
    end
end