#squareSym.c: solution of time-dependent Schroedinger Equation       
#for wavepacket - wavepacket scattering                  
#This is lite version which uses less  memory
#Square potential *and* symmetrization included           
#Copyright 1999, R. Landau, and Jon Maestri, Oregon State University
#supported by: US National Science Foundation, Northwest Alliance
#for Computational Science and Engineering (NACSE),
#US Department of Energy                               

#Commented out a block of fclose statements to alleviate linux woes.

#include <stdio.h>
#include <math.h>
#include <time.h>

define xDim 201            #number space pts, must be odd for simpson

FILE *out1, *out2, *out3a, *out3b, *out3c, *out7a, *out7b, *out7c;
FILE *out8, *out10, *out11, *out12, *out13, *out14, *input;

#Prototypes
#void Potential(); 
void Initialize(void);
void SolveSE(void);
void Output();
void Probability();
#Global Variables                                           
double RePsi[xDim][xDim][2], ImPsi[xDim][xDim][2];
double rho_1[xDim], rho_2[xDim], SumRho[xDim];
#double rho_1[xDim], rho_2[xDim], SumRho[xDim], corr[xDim]; 
#removed J[xDim][xDim],  v[xDim][xDim], Rho[xDim][xDim];               
double w[3];
double a1, a2, a3, a4, alpha, dx, dt, Er, Ei, Eri, Eii;
double k1, k2, m1, m2, dm1, dm2, dm12, dm22, dxx2, dtx, con, con2;
double Ptot, Ptot_i;
double sol, sig, Rho, v, vmax, x01, x02, y;
int choice, symmetry,  Nt, N1, N2, nprint;
time_t timei, timef;
#-------------------------------------------------------------------------
#MAIN

main(){ 
    int i, j;
    time_t timei, timef;
    
    timei = time(NULL);        #start  program timing
    
    #inititialize variables
    input = fopen("in.dat", "r");
    fscanf(input, "%d %lf %lf %lf %lf", &Nt, &k1, &k2, &dx, &dt);
    fscanf(input, "%lf %lf %lf  %lf  %lf", &sig, &x01, &x02, &m1, &m2);
    fscanf(input, "%lf %lf %d %d", &vmax, &alpha, &nprint, &symmetry);
    fclose(input);
    
    #comment out these inline values
    Nt = 40000;  #Number of time steps 
    k1 = 110.;  #wave vectors 
    k2 = 110.;
    dx = 0.002;  #space step  
    dt = 6.0e-8; #time step  
    sig = 0.05;   #wave packet width parameter (Delta x)
    x01 = 0.25;  #Start Initial Positions 
    x02 = 0.75;
    m1 = 1./2.;       #Mass of particles 
    m2 = 5.;
    vmax = -49348.; #Maximum value of Potential 
    alpha = 0.062;   #Potential width parameter 
    nprint = 300; #Print out data every nprint time steps 
    
    #Some constant combos needed in prog 
    N1 = xDim-2; #Max number in sum
    N2 = xDim-2;
    dm1 = 1./m1;
    dm2 = 1./m2;
    dm12 = 0.5/m1;
    dm22 = 0.5/m2;
    dxx2 = dx*dx;
    dtx = dt/dxx2;
    con = -1./(2.*dxx2);
    con2 = (dm1+dm2)*dtx;
    
    #Open files 
    out2 = fopen("V.dat", "wb");
    out8 = fopen("SumRho.dat", "wb");
    out10 = fopen("logP_t.dat", "wb");
    out11 = fopen("E_t.dat", "wb");
    out12 = fopen("logE_t.dat", "wb");
    out14 = fopen("params.dat", "wb");
    
    #not needed for trial runs
    #out3a = fopen("C_0T.dat", "wb"); 
    #out3b = fopen("C_.5T.dat", "wb");
    #out3c = fopen("C_T.dat", "wb");
    #out7a = fopen("Rho_0.dat", "wb");
    #out7b = fopen("Rho_.5.dat", "wb");
    #out7c = fopen("Rho_1.dat", "wb");
    #out13 = fopen("c_x.dat", "wb");
    
    #Choice Menu:
    #choice = 0, view Rho   (3d)                   
    #choice = 1, view rho_1 (2d)                   
    #choice = 2, view corr(x) (2d)  (experimental) 
    #choice = 3, view rho_1 + rho_2                
    choice = 1;
    
    #Symmetry Menu:
    #symmetry = 0, do not impose syymerty          
             #= +1, symmetrize wavefunction         
             #= -1, antisymmetrize wavefunction    
    symmetry
    #Print  initial values to the params file
    fprintf(out14, "choice =  %d\n", choice);
    fprintf(out14, "symmetry =  %d\n", symmetry);
    fprintf(out14, "k1 =  %lf\n", k1);
    fprintf(out14, "k2 =  %lf\n", k2);
    fprintf(out14, "dx =  %lf\n", dx);
    fprintf(out14, "dt =  %e\n", dt);
    fprintf(out14, "sig =  %lf\n", sig);
    fprintf(out14, "alpha =  %lf\n", alpha);
    fprintf(out14, "x01 =  %lf\n", x01);
    fprintf(out14, "x02 =  %lf\n", x02);
    fprintf(out14, "N1 =  %d\n", N1);
    fprintf(out14, "N2 =  %d\n", N2);
    fprintf(out14, "m1 =  %lf\n", m1);
    fprintf(out14, "m2 =  %lf\n", m2);
    fprintf(out14, "vmax =  %lf\n", vmax);
    fprintf(out14, "nprint =  %d\n", nprint);
    fprintf(out14, "xDim =  %d\n", xDim);
    fprintf(out14, "Nt =  %d\n", Nt);
    
    #Initialize Simpson integration weights
    #[end values not used if psi(ends) = 0]
    w[0] = dx/3.; 
    w[1] = 4.*dx/3.;
    w[2] = 2.*dx/3.;
    
    #initialize Potential thru all space 
    Potential();
    
    #initialize  Wave packet 
    Initialize();
    
    #Solve Schrodinger equation
    SolveSE();
    
    #end the time of program 
    timef = time(NULL);
    fprintf(out14, "\nElapsed time is: %ld seconds\n", timef-timei);
     
    fclose(out8);
    fclose(out10);
    fclose(out11);
    fclose(out12);
    fclose(out14);
    }
#-------------------------------------------------------------------------
#Initialize                                                       
void Initialize(){
    double x1, x2, ww, X01, X02;
    double ai1, ai2;
    int i, j, k, p;
    
    #determine omegas in terms of wave vectors 
    ww = (k1*k1/(2.*m1) + k2*k2/(2.*m2))*0.5*dt;
    
    #Initialize  wave function, scale initial positions to box size
    x1 = 0.;
    X01 = x01*xDim*dx;
    X02 = x02*xDim*dx;

    for (i = 1; i <= N1; i++){
    	x1 = x1+dx;
    	x2 = 0.;
        
    	for (j = 1; j <= N2; j++){
    		x2 = x2+dx;
    		y = k1*x1-k2*x2;
    		y = y-ww;
    		a1 = (x1-X01)/sig;
    		a2 = (x2-X02)/sig;
    		a4 = exp(-(a1*a1+a2*a2)/2.);
    		RePsi[i][j][0] = a4* cos(y);
    		ImPsi[i][j][0] = a4* sin(y);
    		}
    	}
    #Set wavefunction to zero on boundary (should never be called anyway)
    #at x2 edges RePsi is zero 
    for (i = 1; i <= N2; i++){
    	RePsi[i][N2+1][0] = 0.;
    	RePsi[i][0][0] = 0.;
    	}	
    
    #at x1 edges RePsi is zero
    for (j = 0; j <= N1+1; j++){
        RePsi[N1+1][j][0] = 0.;
    	RePsi[0][j][0] = 0.;
        }
    
    #Find the initial (unnormalized) energy at roughly t = 0
    Eri = 0.;
    Eii = 0.;
    p = 1;

    for (i = 1; i <= N1; i++){
    	k = 1; 
        
    	if (p == 3)
            p = 1; 
            
    	for (j = 1; j <= N2; j++){
    		if (k == 3)
                k = 1; 
    		#special square well 
            
    		if(alpha >= abs(i-j)*dx)
                {v=vmax;}
            
            else
                {v=0.;}
            
    		a1 = -2*(dm1+dm2+dx*dx*v)*(RePsi[i][j][0]*RePsi[i][j][0]
    		     +ImPsi[i][j][0]*ImPsi[i][j][0]);
            
    		a2 = dm1*(RePsi[i][j][0]*(RePsi[i+1][j][0]+RePsi[i-1][j][0])
    		     +ImPsi[i][j][0]*(ImPsi[i+1][j][0]+ImPsi[i-1][j][0]));
            
    		a3 = dm2*(RePsi[i][j][0]*(RePsi[i][j+1][0]+RePsi[i][j-1][0])
    		    +ImPsi[i][j][0]*(ImPsi[i][j+1][0]+ImPsi[i][j-1][0]));
            
    		ai1 = dm1*(RePsi[i][j][0]*(ImPsi[i+1][j][0]+ImPsi[i-1][j][0])
    		    -ImPsi[i][j][0]*(RePsi[i+1][j][0]+RePsi[i-1][j][0]));
            
    		ai2 = dm2*(RePsi[i][j][0]*(ImPsi[i][j+1][0]+ImPsi[i][j-1][0])
    		    -ImPsi[i][j][0]*(RePsi[i][j+1][0]+RePsi[i][j-1][0]));
            
    		Eri = Eri + w[k]*w[p]*con*(a1 + a2 + a3);
            
    		Eii = Eii + w[k]*w[p]*con*(ai1 + ai2);
            
    		k = k+1; 
    		}
        
    	p = p+1; 
    	}

    #print initial Energy to "params" file 
    fprintf(out14, "E (unnormalized) initial = %lf\n", Eri);
    fprintf(out14, "E imaginary intitial = %lf\n", Eii);
    }
#-------------------------------------------------------------------------
#Probability                                                                 
void Probability(int n){
    double tmp, Prel, x;
    int i,j,k,p;
    
    #initialize the single probability arrays to zero
    for (i = 0; i <= N1+1; i++){
    	rho_1[i] = 0.;
    	rho_2[i] = 0.;
        corr[i]  = 0.;   
        }
    
    #Normalize Rho, important for Correlation functions
    Ptot = 0.;
    Prel = 0.;
    p = 1; 
    
    for (i = 1; i <= N1; i++){
    	k = 1; 
        
    	if (p == 3)
            p = 1; 
            
    	for (j = 1; j <= N2; j++){
    	    Rho = RePsi[i][j][0]*RePsi[i][j][1] +ImPsi[i][j][0]*ImPsi[i][j][0];
            #Impose symmetry or antisymmetry
    		Rho = Rho + symmetry*(RePsi[i][j][0]*RePsi[j][i][1]+ ImPsi[i][j][0]*ImPsi[j][i][0]);
    		
            if (k == 3)
                k = 1; 
                
    		(Ptot) = (Ptot) + w[k]*w[p]*Rho;
            
    		k = k+1; 
    		}
        
    	p = p+1; 
        }
    
    if (n == 1)
        (Ptot_i) = (Ptot);          #Assign  initial 
        
    #Renormalize Rho
    for (i = 1; i <= N1; i++){
            
        for (j = 1; j <= N2; j++){ 
            Rho[i][j] = Rho[i][j]/(Ptot);
            }
        }
    
    #Integrate out 1D probabilites from 2D 
    p = 1;
    
    for (i = 1; i <= N1; i++){  
    	k = 1;
    	if (p == 3)
            p = 1;
            
    	for (j = 1; j <= N2; j++){
    		if (k == 3)
                k = 1; 
    		
            Rho = RePsi[i][j][0]*RePsi[i][j][1]
    				 +ImPsi[i][j][0]*ImPsi[i][j][0];
            #Impose symmetry or antisymmetry
    		Rho = Rho + symmetry*(RePsi[i][j][0]*RePsi[j][i][1]
    					 + ImPsi[i][j][0]*ImPsi[j][i][0]);
    		rho_1[i] = rho_1[i] + w[k]*Rho;
            
            
    		Rho = RePsi[j][i][0]*RePsi[j][i][1]
    				 +ImPsi[j][i][0]*ImPsi[j][i][0];
            #Impose symmetry or antisymmetry
    		Rho = Rho + symmetry*(RePsi[j][i][0]*RePsi[i][j][1]
    					 + ImPsi[j][i][0]*ImPsi[i][j][0]);
    		rho_2[i] = rho_2[i] + w[k]*Rho;
            
    		k = k+1; 
    		}
        
        #sum 1D probabilities and print to file
        SumRho[i] = rho_1[i] + rho_2[i];
        
        if (fabs(SumRho[i]) < 1.e-20) 
            SumRho[i] = 0.;
    
        if (i%10 == 0) 
            fprintf(out8, "%e\n", SumRho[i]);
    
        p = p+1; 
        }
    
    fprintf(out8, "\n");
    
    #find relative probability and print it to file 
    tmp  = fabs(((Ptot)/(Ptot_i ))-1.);
    	
    if (tmp  != 0.) 
        Prel= log10(tmp);
        
    fprintf(out10, "%d %e\n", n, Prel);
    
    #Determine 1 particle correlation function 
    #for i+j fixed (on other diagnol) vs x = i-j 
    
    if (n == Nt/2){
        for (i = 1; i <= N1; i = i+5){ 
            j =  N1+1-i;
            x = i-j;
            x = fabs(x);
            if ((rho_1[i] != 0.)&&(rho_2[j] != 0.))
                corr[i] =  Rho[i][j]/rho_1[i]/rho_2[j];
                
            if ((rho_1[j] != 0.)&&(rho_2[i] != 0.))
                corr[i] =  corr[i]+Rho[j][i]/rho_1[j]/rho_2[i];
                
            if (corr[i] != 0.) 
                corr[i] = log10(fabs(corr[i]));
                
            fprintf(out13, "%lf %e\n", x, corr[i]);
            }
        
        fprintf(out13, "\n"); 
        }
    }
#-------------------------------------------------------------------------
#Potential- now square
void Potential(){
    double tmp;
    int i, j;
    
    tmp = alpha/dx;
    
    for (i = 0; i <= N1+1; i++){
    	for (j = 0; j <= N2+1; j++){
    		 if (abs(i-j)*dx <= alpha) 
                 v[i][j] = vmax;
    		 else 
                 v[i][j] = 0.;
    		} 
        }
    }

#-------------------------------------------------------------------------
#SolveSE                                                              
void SolveSE(){
    #solve time dependent Schroedinger equation 
    int n, i, j;
    char s[] = "ru21.0001";
    FILE *out;

    #Start of the loop that finds the probability at each time step
    for(n = 1; n <= Nt;n++){
    	#compute real part of  WF
    	for (i = 1; i <= N1; i++){
    		for (j = 1; j <= N2; j++){
                #special square well 
    		    if (alpha>=abs(i-j)*dx)
                    {v=vmax;}
                
                else
                    {v=0.;}
                
    			a2 = dt*v*ImPsi[i][j][0]+con2*ImPsi[i][j][0];
                
    			a1 = dm12*(ImPsi[i+1][j][0]+ImPsi[i-1][j][0])
    			   +dm22*(ImPsi[i][j+1][0]+ImPsi[i][j-1][0]);
                   
    			RePsi[i][j][1] = RePsi[i][j][0]-dtx*a1+2.*a2;
    
    
        		#compute probability density (old, now Rho inlined)
    			#if (n%(nprint) == 0|n == 1|n == Nt/2|n == Nt)
    			#Rho[i][j] = RePsi[i][j][0]*RePsi[i][j][1]
    				 #+ImPsi[i][j][0]*ImPsi[i][j][0];       
    			}
    		}
    
        #Integrate out all the different Probabilities
    	if (n%(nprint) == 0|n == 1|n == Nt/2|n == Nt) 
            Probability(n);
    
    	#imaginary part of wave packet is next
    	for (i = 1; i <= N1; i++){
                
    		for (j = 1; j <= N2; j++){      
    		    #special square well  
                
    		    if (alpha>=abs(i-j)*dx)
                    {v=vmax;}
                
                else
                    {v=0.;}
                
    			a2 = dt*v*RePsi[i][j][1]+con2*RePsi[i][j][1];
                
    			a1 = dm12*(RePsi[i+1][j][1]+RePsi[i-1][j][1])
    			    +dm22*(RePsi[i][j+1][1]+RePsi[i][j-1][1]);
                    
    			ImPsi[i][j][1] = ImPsi[i][j][0]+dtx*a1-2.*a2 ;
    			}
    		}
    
        #Find  Energy
    	if (n%(nprint) == 0|n == 1) 
            Energy(n);
    
        #Generate data files (most of them)
        if (n%(nprint) == 0|n ==  1|n == Nt/2|n == Nt) 
            Output(n);
    		
        #new iterations are now the old ones, recycle 
        for (i = 1; i <= N1; i++){
                
    	    for (j = 1; j <= N2; j++){
    		    ImPsi[i][j][0] = ImPsi[i][j][1];
    		    RePsi[i][j][0] = RePsi[i][j][1]; 
    		    }
    		}
        }
    }
#-------------------------------------------------------------------------
#Energy                                                              
void Energy(int n){
    double ai1, ai2, tmp, Erel;
    int i,j,k,p;

    #calculate total energy of system 
    Er = 0.;
    Ei = 0.;
    p = 1;      #wf zero at boundaries, therefore no p,k = 0 simpson terms */

    for (i = 1; i <= N1; i++){
    	k = 1;
    	
        if (p == 3)
            p = 1;
        
        for (j = 1; j <= N2; j++){
    		
            if (k == 3)
                k = 1;
    		#square well in line
            
    		if(alpha >= abs(i-j)*dx)
                {v=vmax;}
                
            else
                {v=0.;}
            
    		a1 = -2*(dm1+dm2+dx*dx*v)*(RePsi[i][j][1]*RePsi[i][j][1]
    		    +ImPsi[i][j][1]*ImPsi[i][j][1]);
            
    		a2 = dm1*(RePsi[i][j][1]*(RePsi[i+1][j][1]+RePsi[i-1][j][1])
    		    +ImPsi[i][j][1]*(ImPsi[i+1][j][1]+ImPsi[i-1][j][1]));
            
    		a3 = dm2*(RePsi[i][j][1]*(RePsi[i][j+1][1]+RePsi[i][j-1][1])
    		    +ImPsi[i][j][1]*(ImPsi[i][j+1][1]+ImPsi[i][j-1][1]));
            
    		ai1 = dm1*(RePsi[i][j][1]*(ImPsi[i+1][j][1]+ImPsi[i-1][j][1])
    		     -ImPsi[i][j][1]*(RePsi[i+1][j][1]+RePsi[i-1][j][1]));
            
    		ai2 = dm2*(RePsi[i][j][1]*(ImPsi[i][j+1][1]+ImPsi[i][j-1][1])
    		     -ImPsi[i][j][1]*(RePsi[i][j+1][1]+RePsi[i][j-1][1]));
            
    		Er = Er + w[k]*w[p]*con*(a1 + a2 + a3); 
            
    		Ei = Ei + w[k]*w[p]*con*(ai1 + ai2);
            
    		k = k+1;
    		}
        
    	p = p+1;
        }

    #Normalize 
    Er = Er/Ptot_i;  
    
    Ei = Ei/Ptot_i;
    
    #make sure number is not too small, gnuplot can't take them 
    if (fabs(Er) <= 1.e-20)
        Er = 0.;
    
    #print to "EvsT.dat" file 
    fprintf(out11, "%d %e\n", n, Er);
    
    #find relative Energy 
    tmp  = fabs((Er/Eri)-1.);
    
    if (tmp  != 0.)
        Erel = log10(tmp);
    	
    #print to "ErelvsT.dat"
    fprintf(out12, "%d %e\n", n, Erel);
    
    return;
    }
#-------------------------------------------------------------------------
#Output
void Output(n){
    int i, j;
    char s[] = "run.00001";
    
    #Create a unique file name with data for each time step
    if (n < 10){
        s[8] =  n+48;
        }
    
    if (n < 100 && n > 9){
        s[7] = (n/10)+48;
    	s[8] =  (n%10)+48;
    	}
    
    if (n < 1000 && n > 99){
    	s[6] = (n/100)+48;
       	s[7] =  (((n%100))/10)+48;
    	s[8] =  (n%10)+48;
    	}
    
    if (n < 10000 && n > 999){
        s[5] = (n/1000)+48;
    	s[6] = (((n%1000))/100)+48;
    	s[7] =  (((n%100))/10)+48;
    	s[8] =  (n%10)+48;
    	}
    
    if (n < 100000 && n > 9999){
    	s[4] = (n/10000)+48;
    	s[5] = (((n%10000))/1000)+48;
    	s[6] = (((n%1000))/100)+48;
    	s[7] =  (((n%100))/10)+48;
    	s[8] =  (n%10)+48;
    	}
    
    #Print probability vs position data at each time
    if (choice == 0){
        if (n == 1){
            out1 = fopen(s,"wb");
            for (i = 1; i <= N2; i++){
    		    for (j = 1; j <= N2; j++){
    			     Rho = RePsi[i][j][0]*RePsi[i][j][1]+ImPsi[i][j][0]*ImPsi[i][j][0];
                     #Impose symmetry or antisymmetry
    		        Rho = Rho + symmetry*(RePsi[i][j][0]*RePsi[j][i][1]+ImPsi[i][j][0]*ImPsi[j][i][0]);
    			    if (Rho < 1.e-20)
                        Rho = 1.e-20;
    			    fprintf(out1, "%e\n", Rho);
    			    }
    		    fprintf(out1, "\n");
        		}
    	    fclose(out1);
            }
        
        out1 = fopen(s,"wb");
    
    	for (i = 1; i <= N2; i++){
    		for (j = 1; j <= N2; j++){
    			Rho = RePsi[i][j][0]*RePsi[i][j][1]+ImPsi[i][j][0]*ImPsi[i][j][0];
                #Impose symmetry or antisymmetry 
    		    Rho = Rho + symmetry*(RePsi[i][j][0]*RePsi[j][i][1]+ImPsi[i][j][0]*ImPsi[j][i][0]);
    			if (Rho < 1.e-20)
                    Rho = 1.e-20;
    			fprintf(out1, "%e\n", Rho);
    			}
    		fprintf(out1, "\n");
    		}
        
    	fclose(out1);
        }
    
    if (choice == 1){
        out1 = fopen(s,"wb");
    	for (i = 1; i <= N1; i++){
    		if (rho_1[i] < 1.e-20)
                rho_1[i] = 1.e-20;
    		if (rho_2[i] < 1.e-20)
                rho_2[i] = 1.e-20;
    		fprintf(out1, "%d %e %e\n", i, rho_1[i], rho_2[i]);
    		}
    	fclose(out1);
    	}
    
    if (choice == 2){
        out1 = fopen(s,"wb");
    	for (i = 1; i <= N2; i++){
    		if (rho_1[i] < 1.e-20)
                rho_1[i] = 1.e-20;
    		if (rho_2[i] < 1.e-20)
                rho_2[i] = 1.e-20;
    		fprintf(out1, "%d %e %e\n", i, corr[i]);
    		}
    	fclose(out1);
    	}
    
    if (choice == 3){
        out1 = fopen(s,"wb");
    	for (i = 1; i <= N1; i++){
    		sol = rho_2[i]+rho_1[i];
    		if (sol < 1.e-20)
                sol = 1.e-20;
    		fprintf(out1, "%d %e\n", i, sol);
    		}
    	fclose(out1);
    	}
    
    #Print out some data sets for time = 1 
    
    if (n == 1){
        #print out potential
    	for (i = 1; i <= N1; i = i+10){
    		for (j = 1; j <= N2; j = j+10){
    		    #special square well
    		    if (alpha>=abs(i-j)*dx)
                    {v=vmax;}
                else
                    {v=0.;}
    			fprintf(out2, "%e\n", v);
    			}
    		fprintf(out2, "\n");
    		}
    	 
    	#Print out two particle probability & correlation at t = 1 
    	#need open files here for some reason!      
        out7a = fopen("Rho_0.dat", "wb");
    	out3a = fopen("C_0T.dat", "wb");
        
    	for (i = 1; i <= N1; i = i+10){
    		for (j = 1; j <= N2; j = j+10){
    		    Rho = RePsi[i][j][0]*RePsi[i][j][1]+ImPsi[i][j][0]*ImPsi[i][j][0];
    			sol = Rho;
                #Impose symmetry or antisymmetry 
    		    Rho = Rho + symmetry*(RePsi[i][j][0]*RePsi[j][i][1]+ImPsi[i][j][0]*ImPsi[j][i][0]);
    			if (sol < 1.2e-20)
                    sol = 0.;
    			fprintf(out7a, "%e\n", sol);
    			if ((rho_1[i]  != 0.) && (rho_2[j]  != 0.)){
    				sol = log10(fabs(Rho/rho_1[i]/rho_2[j]));
                    }
    			else 
                    sol = 0.;
    			fprintf(out3a, "%e\n", sol);
    			}
    		fprintf(out7a, "\n");
    		fprintf(out3a, "\n");
    		}
        }
    
    #print out a couple more two particle probabilities
    if (n == (Nt/2)){
    	out7b = fopen("Rho_.5.dat", "wb");
    	out3b = fopen("C_.5T.dat", "wb");
    	for (i = 1; i <= N1; i = i+10){
    		for (j = 1; j <= N2; j = j+10){
    		    Rho = RePsi[i][j][0]*RePsi[i][j][1]+ImPsi[i][j][0]*ImPsi[i][j][0];
                #Impose symmetry or antisymmetry 
    		    Rho = Rho + symmetry*(RePsi[i][j][0]*RePsi[j][i][1]+ImPsi[i][j][0]*ImPsi[j][i][0]);
    			sol = Rho;
    			if (sol < 1.2e-20)
                    sol = 0.;
    			fprintf(out7b, "%e\n", sol);
    			if ((rho_1[i]  != 0.) && (rho_2[j]  != 0.)){
    			   sol = log10(fabs(Rho/rho_1[i]/rho_2[j]));    
                   }
    			else 
                    sol = 0.;
    			fprintf(out3b, "%e\n", sol);
    			}
    		fprintf(out7b, "\n");
    		fprintf(out3b, "\n");
    		}
        }
    
    if (n == Nt){
        out7c = fopen("Rho_1.dat", "wb");
    	out3c = fopen("C_T.dat", "wb");
        
    	for (i = 1; i <= N1; i = i+10){
    		for (j = 1; j <= N2; j = j+10){
    		    Rho = RePsi[i][j][0]*RePsi[i][j][1]+ImPsi[i][j][0]*ImPsi[i][j][0];
                #Impose symmetry or antisymmetry 
    		    Rho = Rho+symmetry*(RePsi[i][j][0]*RePsi[j][i][1]+ImPsi[i][j][0]*ImPsi[j][i][0]);
    			sol = Rho;
    			if (sol < 1.2e-20)
                    sol = 0.;
    			fprintf(out7c, "%e\n", sol);
    			if ((rho_1[i]  != 0.) && (rho_2[j]  != 0.)){
    			   sol = log10(fabs(Rho/rho_1[i]/rho_2[j]));    
                   }
    			else 
                    sol = 0.;
    			fprintf(out3c, "%e\n", sol);
    			}
    		fprintf(out7c, "\n");
    		fprintf(out3c, "\n");
    		}
        }
    
    fclose(out7c);
    fclose(out3c); 
    fclose(out7a);
    fclose(out7b);
    fclose(out3a);
    fclose(out3b);
    fclose(out3c);
    fclose(out2); kirk  
    }