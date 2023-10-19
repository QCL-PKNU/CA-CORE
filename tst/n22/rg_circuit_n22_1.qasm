OPENQASM 2.0;
include "qelib1.inc";
qreg q[22];
creg c[22];
rz(-pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-1.9044363664374213) q[1];
sx q[1];
rz(6.832200689828058) q[1];
sx q[1];
rz(11.3292143272068) q[1];
rz(0.02020740913621762) q[1];
rz(pi/4) q[3];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(6.07975176994234) q[6];
id q[6];
rz(pi/2) q[6];
sx q[6];
rz(4.860742341425925) q[6];
sx q[6];
rz(5*pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi) q[7];
rz(pi) q[7];
x q[7];
x q[8];
cx q[1],q[8];
rz(-0.02020740913621762) q[8];
cx q[1],q[8];
rz(pi) q[1];
rz(pi/2) q[1];
rz(-pi/2) q[1];
rz(0.02020740913621762) q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
rz(1.7840737074778499) q[11];
sx q[11];
rz(1.653884325479613) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(6.0576911170767165) q[11];
rz(pi/2) q[11];
sx q[12];
cx q[9],q[12];
rz(0) q[12];
sx q[12];
rz(8.988854623702547) q[12];
sx q[12];
rz(3*pi) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[8];
cx q[8],q[12];
rz(4.2530125947946384) q[12];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
x q[9];
cx q[13],q[4];
rz(4.4210899954601945) q[4];
cx q[13],q[4];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/4) q[4];
cx q[4],q[7];
rz(0.16342095362829312) q[4];
cx q[12],q[4];
rz(-4.2530125947946384) q[4];
sx q[4];
rz(2.9112715310521944) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[12],q[4];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(0) q[4];
sx q[4];
rz(3.371913776127392) q[4];
sx q[4];
rz(13.514369601935725) q[4];
sx q[4];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
cx q[15],q[10];
cx q[10],q[15];
cx q[15],q[10];
rz(3.4559693062543255) q[10];
sx q[10];
rz(1.8919973011826186) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(0) q[15];
sx q[15];
rz(5.435344613469507) q[15];
sx q[15];
rz(3*pi) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[5],q[16];
cx q[16],q[5];
cx q[5],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(1.4365829739741847) q[16];
rz(5.205125511675727) q[5];
rz(pi/2) q[5];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/4) q[17];
cx q[2],q[17];
rz(-pi/4) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(-pi/2) q[17];
rz(0.4487757834804914) q[17];
rz(4.370015729204768) q[17];
rz(1.9661419897434) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(2.981490991828404) q[2];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[3],q[18];
rz(-pi/4) q[18];
cx q[3],q[18];
rz(pi/4) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
x q[18];
rz(-pi/2) q[18];
rz(5.453203645790947) q[3];
rz(-pi/2) q[3];
cx q[3],q[2];
rz(-2.981490991828404) q[2];
cx q[3],q[2];
rz(0) q[2];
sx q[2];
rz(8.985738738736135) q[2];
sx q[2];
rz(3*pi) q[2];
rz(-pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
x q[19];
rz(-pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[19],q[5];
rz(0) q[19];
sx q[19];
rz(1.22592984967374) q[19];
sx q[19];
rz(3*pi) q[19];
rz(0) q[5];
sx q[5];
rz(1.22592984967374) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[19],q[5];
rz(-pi/2) q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
rz(4.665189574397658) q[19];
rz(-pi/2) q[5];
rz(-5.205125511675727) q[5];
cx q[5],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi) q[15];
rz(0) q[15];
sx q[15];
rz(4.558652337734831) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[5],q[19];
cx q[19],q[5];
sx q[19];
cx q[19],q[2];
cx q[2],q[19];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[5],q[15];
rz(0) q[15];
sx q[15];
rz(1.7245329694447558) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[5],q[15];
rz(pi/2) q[15];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[20],q[0];
rz(5.344864377526088) q[0];
cx q[20],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
cx q[0],q[16];
rz(-1.4365829739741847) q[16];
cx q[0],q[16];
rz(1.444204302881609) q[0];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
rz(0.5027866242981385) q[20];
cx q[17],q[20];
rz(-4.370015729204768) q[20];
sx q[20];
rz(0.5330971064308114) q[20];
sx q[20];
rz(3*pi) q[20];
cx q[17],q[20];
rz(pi) q[17];
rz(0) q[17];
sx q[17];
rz(4.867958728650345) q[17];
sx q[17];
rz(3*pi) q[17];
rz(0) q[20];
sx q[20];
rz(5.750088200748775) q[20];
sx q[20];
rz(13.292007065676009) q[20];
cx q[20],q[16];
rz(2.7723586092560284) q[20];
cx q[8],q[17];
rz(0) q[17];
sx q[17];
rz(1.4152265785292413) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[8],q[17];
rz(1.487361882871706) q[17];
cx q[17],q[8];
rz(-1.487361882871706) q[8];
cx q[17],q[8];
rz(2.576726507191291) q[17];
rz(1.487361882871706) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[21],q[14];
rz(1.8316053568156814) q[14];
cx q[21],q[14];
rz(0.9911111785408171) q[14];
rz(pi/2) q[14];
cx q[13],q[14];
rz(0) q[13];
sx q[13];
rz(1.4082884646046014) q[13];
sx q[13];
rz(3*pi) q[13];
rz(0) q[14];
sx q[14];
rz(1.4082884646046014) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[13],q[14];
rz(-pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(-pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-3.58650939665171) q[13];
rz(pi/2) q[13];
rz(-pi/2) q[14];
rz(-0.9911111785408171) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[10];
rz(0.6438802371839447) q[10];
cx q[14],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(0.3312661036449071) q[10];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[16],q[14];
rz(5.484811039776404) q[14];
cx q[16],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/4) q[14];
cx q[14],q[6];
rz(0) q[16];
sx q[16];
rz(4.669736877027281) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[18],q[10];
rz(-0.3312661036449071) q[10];
cx q[18],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-pi/4) q[10];
id q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[18],q[7];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[3],q[13];
rz(0) q[13];
sx q[13];
rz(0.8435521225506109) q[13];
sx q[13];
rz(3*pi) q[13];
rz(0) q[3];
sx q[3];
rz(5.439633184628976) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[3],q[13];
rz(-pi/2) q[13];
rz(3.58650939665171) q[13];
rz(1.2072270007232409) q[13];
rz(pi/2) q[13];
rz(-pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
cx q[15],q[3];
x q[15];
rz(pi) q[15];
x q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[2],q[3];
rz(3.1107674501643787) q[3];
cx q[2],q[3];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(5.281818856474507) q[2];
rz(pi/2) q[2];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(2.0243279277340167) q[3];
rz(-pi/4) q[6];
cx q[14],q[6];
rz(-pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[13];
rz(0) q[13];
sx q[13];
rz(0.8399504133369331) q[13];
sx q[13];
rz(3*pi) q[13];
rz(0) q[14];
sx q[14];
rz(0.8399504133369331) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[14],q[13];
rz(-pi/2) q[13];
rz(-1.2072270007232409) q[13];
rz(pi/2) q[13];
rz(pi/4) q[13];
rz(-pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[5],q[14];
rz(2.7095755954748184) q[14];
cx q[5],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/4) q[14];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(0.8240231672281918) q[5];
sx q[5];
rz(7.323979409787228) q[5];
sx q[5];
rz(10.32396917245634) q[5];
rz(-5.134673860553782) q[5];
rz(pi/2) q[5];
rz(pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[7],q[18];
cx q[6],q[7];
rz(-1.3340211795366521) q[6];
rz(pi/2) q[6];
cx q[9],q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(0.07545188124672825) q[21];
x q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[12],q[21];
rz(4.918173249765428) q[21];
cx q[12],q[21];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(1.7842428989455192) q[21];
cx q[12],q[21];
rz(-1.7842428989455192) q[21];
cx q[12],q[21];
rz(5.552680663939243) q[12];
sx q[12];
rz(8.493557718938902) q[12];
sx q[12];
rz(14.517914163829928) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[2];
rz(0) q[12];
sx q[12];
rz(2.987351178575407) q[12];
sx q[12];
rz(3*pi) q[12];
rz(0) q[2];
sx q[2];
rz(2.987351178575407) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[12],q[2];
rz(-pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(0) q[12];
sx q[12];
rz(7.391225378143816) q[12];
sx q[12];
rz(3*pi) q[12];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi/2) q[2];
rz(-5.281818856474507) q[2];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(1.025442329031427) q[21];
rz(pi) q[21];
rz(3.3838039548315098) q[21];
rz(0) q[21];
sx q[21];
rz(8.769730239159955) q[21];
sx q[21];
rz(3*pi) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(3.9404984089002086) q[9];
cx q[9],q[0];
rz(-3.9404984089002086) q[0];
sx q[0];
rz(1.04897968207327) q[0];
sx q[0];
rz(3*pi) q[0];
cx q[9],q[0];
rz(0) q[0];
sx q[0];
rz(5.234205625106316) q[0];
sx q[0];
rz(11.921072066787978) q[0];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[11];
rz(0) q[0];
sx q[0];
rz(0.7386625641025) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[11];
sx q[11];
rz(0.7386625641025) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[0],q[11];
rz(-pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/2) q[11];
rz(-6.0576911170767165) q[11];
rz(pi/2) q[11];
sx q[11];
rz(9.094256428091782) q[11];
sx q[11];
rz(5*pi/2) q[11];
rz(3.2631643952078835) q[11];
cx q[11],q[17];
rz(-3.2631643952078835) q[17];
sx q[17];
rz(1.445418254777045) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[11],q[17];
cx q[11],q[19];
rz(0) q[17];
sx q[17];
rz(4.837767052402541) q[17];
sx q[17];
rz(10.111215848785973) q[17];
cx q[19],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[19],q[14];
rz(-pi/4) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/2) q[14];
rz(pi) q[14];
x q[14];
rz(-0.21370956236134542) q[14];
sx q[14];
rz(8.725512005285442) q[14];
sx q[14];
rz(9.638487523130724) q[14];
rz(pi/2) q[14];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(2.5435105604223978) q[9];
cx q[20],q[9];
rz(-2.7723586092560284) q[9];
sx q[9];
rz(1.0739937224377707) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[20],q[9];
cx q[20],q[16];
rz(0) q[16];
sx q[16];
rz(1.613448430152305) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[20],q[16];
cx q[16],q[1];
rz(0.566336962864033) q[1];
cx q[16],q[1];
rz(0.8366753487497336) q[1];
sx q[1];
rz(9.218524690925054) q[1];
sx q[1];
rz(8.588102612019647) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[13],q[1];
rz(-pi/4) q[1];
cx q[13],q[1];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
x q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[1];
rz(-pi/2) q[13];
rz(pi) q[13];
x q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-pi/2) q[16];
cx q[16],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(0.7021866999786529) q[15];
sx q[15];
rz(6.917482007114844) q[15];
sx q[15];
rz(11.633056127980508) q[15];
rz(-pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[2],q[1];
rz(-pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[1];
rz(2.990697735416108) q[1];
rz(3.9942247879494666) q[1];
rz(0) q[1];
sx q[1];
rz(6.31206527868559) q[1];
sx q[1];
rz(3*pi) q[1];
rz(pi) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(3.8464742108192693) q[2];
sx q[2];
rz(4.955566585003839) q[2];
sx q[2];
rz(11.987865451958493) q[2];
rz(0) q[2];
sx q[2];
rz(3.345653017869806) q[2];
sx q[2];
rz(3*pi) q[2];
rz(pi/4) q[20];
cx q[20],q[0];
rz(-pi/4) q[0];
cx q[20],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[6];
rz(0) q[0];
sx q[0];
rz(3.572892784337207) q[0];
sx q[0];
rz(3*pi) q[0];
rz(5.670180454045062) q[20];
sx q[20];
rz(4.728455093645393) q[20];
sx q[20];
rz(11.931270427553832) q[20];
rz(4.798225871216605) q[20];
cx q[21],q[13];
rz(5.858344841621449) q[13];
cx q[21],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
sx q[21];
cx q[13],q[21];
x q[13];
rz(pi/2) q[13];
sx q[13];
rz(4.284710122842181) q[13];
sx q[13];
rz(5*pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(1.942991984664042) q[13];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(0) q[6];
sx q[6];
rz(2.710292522842379) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[0],q[6];
rz(-pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(0.204000255969764) q[0];
sx q[0];
rz(5.96581814136003) q[0];
sx q[0];
rz(11.035978369401409) q[0];
cx q[20],q[0];
rz(1.2994338064215871) q[0];
cx q[20],q[0];
sx q[0];
rz(pi) q[20];
x q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(3.04173354575655) q[20];
rz(-pi/2) q[6];
rz(1.3340211795366521) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(0) q[9];
sx q[9];
rz(5.2091915847418155) q[9];
sx q[9];
rz(9.65362600960301) q[9];
rz(pi/2) q[9];
cx q[9],q[4];
rz(1.0766208108134891) q[4];
cx q[18],q[4];
rz(-1.0766208108134891) q[4];
cx q[18],q[4];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[10],q[18];
rz(0.576803940633785) q[18];
cx q[10],q[18];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
x q[18];
rz(pi/2) q[18];
rz(0) q[4];
sx q[4];
rz(5.315200741812207) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[7],q[4];
rz(0) q[4];
sx q[4];
rz(0.9679845653673791) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[7],q[4];
rz(3.1602198548425173) q[4];
rz(4.73895944053822) q[4];
rz(2.3486434218419663) q[4];
sx q[4];
rz(4.3606291760082) q[4];
sx q[4];
rz(9.508339377389804) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
sx q[7];
x q[9];
cx q[9],q[8];
rz(5.9449887843596265) q[8];
cx q[9],q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(3.9366880034397824) q[8];
cx q[8],q[10];
cx q[10],q[8];
rz(-3.109462830996888) q[10];
sx q[10];
rz(5.202279339856107) q[10];
sx q[10];
rz(12.534240791766267) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[16],q[8];
rz(pi/4) q[16];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[16],q[8];
rz(-pi/4) q[8];
cx q[16],q[8];
rz(-5.082976446048285) q[16];
rz(pi/2) q[16];
cx q[15],q[16];
rz(0) q[15];
sx q[15];
rz(6.152522350417227) q[15];
sx q[15];
rz(3*pi) q[15];
rz(0) q[16];
sx q[16];
rz(0.13066295676235962) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[15],q[16];
rz(-pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(-pi/2) q[16];
rz(5.082976446048285) q[16];
cx q[16],q[4];
rz(pi/4) q[16];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[16],q[4];
rz(-pi/4) q[4];
cx q[16],q[4];
rz(1.5392205474021712) q[16];
rz(pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(2.128831330684782) q[4];
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi) q[8];
rz(-4.469596622985022) q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/4) q[9];
cx q[17],q[9];
cx q[3],q[17];
rz(-2.0243279277340167) q[17];
cx q[3],q[17];
rz(2.0243279277340167) q[17];
sx q[17];
cx q[18],q[17];
cx q[17],q[10];
rz(0.9664823143630035) q[10];
cx q[17],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(5.1301306133963775) q[10];
sx q[10];
rz(6.85492683738644) q[10];
sx q[10];
rz(14.8391855717374) q[10];
rz(-pi/2) q[10];
x q[17];
cx q[17],q[20];
x q[18];
sx q[18];
rz(-3.04173354575655) q[20];
cx q[17],q[20];
rz(3.5471602723580347) q[17];
sx q[17];
rz(6.5946326649720675) q[17];
sx q[17];
rz(13.783743938776432) q[17];
rz(pi/4) q[20];
rz(-pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[5];
rz(0) q[3];
sx q[3];
rz(4.99036000026183) q[3];
sx q[3];
rz(3*pi) q[3];
rz(0) q[5];
sx q[5];
rz(1.2928253069177564) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[3],q[5];
rz(-pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[5];
rz(5.134673860553782) q[5];
rz(-pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[3],q[5];
rz(0.9437415405167789) q[5];
cx q[3],q[5];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(-0.0946658817232282) q[3];
sx q[3];
rz(3.3763979014702645) q[3];
sx q[3];
rz(9.519443842492608) q[3];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/4) q[5];
rz(pi/4) q[5];
rz(1.742013944626441) q[5];
cx q[6],q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[3],q[18];
rz(2.7548552197512173) q[18];
cx q[3],q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/4) q[18];
cx q[18],q[21];
rz(-pi/4) q[21];
cx q[18],q[21];
rz(pi/4) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
x q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[6],q[15];
rz(2.2115632053986216) q[15];
cx q[6],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(-pi/4) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(-pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[11],q[9];
rz(2.4436725895526994) q[9];
cx q[11],q[9];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[7],q[11];
rz(0.014314383954637026) q[11];
cx q[7],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
cx q[14],q[11];
x q[14];
rz(5.659998891995641) q[14];
rz(-pi/2) q[14];
cx q[4],q[11];
rz(-2.128831330684782) q[11];
cx q[4],q[11];
rz(2.128831330684782) q[11];
x q[11];
rz(pi/4) q[11];
rz(0.8107868894721458) q[4];
rz(1.4768018885108305) q[4];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[19];
rz(0.19669790155517278) q[19];
cx q[9],q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
cx q[19],q[0];
rz(pi) q[0];
x q[0];
cx q[0],q[2];
x q[19];
rz(3.4228998002051574) q[19];
rz(pi/2) q[19];
cx q[12],q[19];
rz(0) q[12];
sx q[12];
rz(1.1838608489384077) q[12];
sx q[12];
rz(3*pi) q[12];
rz(0) q[19];
sx q[19];
rz(1.1838608489384077) q[19];
sx q[19];
rz(3*pi) q[19];
cx q[12],q[19];
rz(-pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
cx q[12],q[10];
rz(pi/2) q[10];
rz(1.4774050959891698) q[12];
rz(2.329380019678923) q[12];
cx q[12],q[5];
rz(-pi/2) q[19];
rz(-3.4228998002051574) q[19];
rz(-pi/2) q[19];
cx q[15],q[19];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[10],q[15];
rz(0) q[10];
sx q[10];
rz(8.001563903384255) q[10];
sx q[10];
rz(3*pi) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[19];
rz(-pi/4) q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/4) q[19];
cx q[18],q[19];
rz(0.6730682232720236) q[18];
rz(-pi/4) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(-pi/2) q[19];
rz(1.422161988427341) q[19];
sx q[19];
rz(4.597255973778496) q[19];
sx q[19];
rz(11.970581045530421) q[19];
rz(-0.6610853855128112) q[19];
rz(pi/2) q[19];
rz(0) q[2];
sx q[2];
rz(2.93753228930978) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[0],q[2];
rz(pi/2) q[0];
rz(-2.0197095397457767) q[0];
id q[2];
cx q[2],q[17];
cx q[17],q[2];
cx q[2],q[17];
rz(0) q[17];
sx q[17];
rz(5.1674013144134205) q[17];
sx q[17];
rz(3*pi) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(5.035513242159903) q[2];
rz(5.548231946206889) q[2];
rz(-2.329380019678923) q[5];
sx q[5];
rz(2.992762834579674) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[12],q[5];
rz(-0.11851632644358379) q[12];
rz(0) q[5];
sx q[5];
rz(3.290422472599912) q[5];
sx q[5];
rz(10.01214403582186) q[5];
rz(3.6689639743565623) q[5];
rz(-2.2121407500083428) q[5];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[7],q[9];
rz(4.716794440074424) q[9];
cx q[7],q[9];
rz(-pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[8];
rz(0) q[7];
sx q[7];
rz(6.118520069241919) q[7];
sx q[7];
rz(3*pi) q[7];
rz(0) q[8];
sx q[8];
rz(0.16466523793766763) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[7],q[8];
rz(-pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
cx q[7],q[6];
cx q[6],q[7];
cx q[7],q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[3],q[6];
rz(2.2334420453600132) q[6];
cx q[3],q[6];
rz(0.4168038353785012) q[3];
cx q[3],q[15];
rz(-0.4168038353785012) q[15];
cx q[3],q[15];
rz(0.4168038353785012) q[15];
rz(0) q[15];
sx q[15];
rz(4.89742491972584) q[15];
sx q[15];
rz(3*pi) q[15];
rz(-pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(-0.8122889724450846) q[3];
sx q[3];
rz(6.575301107716401) q[3];
sx q[3];
rz(10.237066933214464) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(5.919361865489784) q[6];
rz(3.174706825874191) q[6];
cx q[6],q[0];
rz(-3.174706825874191) q[0];
sx q[0];
rz(2.3167902228421386) q[0];
sx q[0];
rz(3*pi) q[0];
cx q[6],q[0];
rz(0) q[0];
sx q[0];
rz(3.9663950843374476) q[0];
sx q[0];
rz(14.619194326389348) q[0];
rz(-0.07981792930531384) q[6];
cx q[2],q[6];
rz(-5.548231946206889) q[6];
sx q[6];
rz(1.564579885155418) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[2],q[6];
rz(0) q[6];
sx q[6];
rz(4.718605422024169) q[6];
sx q[6];
rz(15.052827836281583) q[6];
cx q[6],q[3];
rz(1.8816503988477513) q[3];
cx q[6],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(0) q[3];
sx q[3];
rz(6.327074548538047) q[3];
sx q[3];
rz(3*pi) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[20],q[7];
rz(-pi/4) q[7];
cx q[20],q[7];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[11],q[20];
rz(-pi/4) q[20];
cx q[11],q[20];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[11],q[1];
rz(3.2116918998860644) q[1];
cx q[11],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(0) q[1];
sx q[1];
rz(5.038858290292603) q[1];
sx q[1];
rz(3*pi) q[1];
rz(1.0164397310645052) q[1];
sx q[1];
rz(6.983472380539226) q[1];
sx q[1];
rz(11.83379914568186) q[1];
rz(-2.932381708159806) q[1];
sx q[1];
rz(3.450923685429544) q[1];
sx q[1];
rz(12.357159668929185) q[1];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[20],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/4) q[20];
cx q[20],q[10];
rz(-pi/4) q[10];
cx q[20],q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(1.149439789686526) q[10];
rz(pi/2) q[10];
cx q[15],q[10];
rz(0) q[10];
sx q[10];
rz(1.1722218877710804) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[15];
sx q[15];
rz(1.1722218877710804) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[15],q[10];
rz(-pi/2) q[10];
rz(-1.149439789686526) q[10];
sx q[10];
rz(-pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(1.3834020700014662) q[15];
cx q[15],q[6];
cx q[20],q[17];
rz(0.9426501355430958) q[17];
cx q[20],q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi) q[20];
x q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(-1.3834020700014662) q[6];
cx q[15],q[6];
rz(1.3834020700014662) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[13];
rz(-1.942991984664042) q[13];
cx q[7],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-3.850788555874882) q[13];
sx q[13];
rz(7.241343414826061) q[13];
sx q[13];
rz(13.275566516644261) q[13];
rz(0.4832959992261902) q[13];
rz(pi/2) q[13];
rz(4.659167840586854) q[7];
sx q[7];
rz(3.6262915851209074) q[7];
sx q[7];
rz(14.484899003648342) q[7];
rz(pi/4) q[7];
rz(-pi/2) q[8];
rz(4.469596622985022) q[8];
cx q[16],q[8];
rz(-1.5392205474021712) q[8];
cx q[16],q[8];
rz(-0.44332310411800524) q[16];
cx q[4],q[16];
rz(-1.4768018885108305) q[16];
sx q[16];
rz(0.8700504487752405) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[4],q[16];
rz(0) q[16];
sx q[16];
rz(5.413134858404346) q[16];
sx q[16];
rz(11.344902953398215) q[16];
rz(4.020494364276059) q[16];
cx q[16],q[12];
rz(-4.020494364276059) q[12];
sx q[12];
rz(2.657044767978265) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[16],q[12];
rz(0) q[12];
sx q[12];
rz(3.6261405392013213) q[12];
sx q[12];
rz(13.563788651489023) q[12];
rz(-pi/2) q[16];
cx q[0],q[16];
sx q[0];
rz(1.6281197540185781) q[0];
sx q[0];
rz(5.353103462391651) q[0];
rz(0.30676364217002106) q[0];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(0.8809152163216759) q[16];
sx q[16];
rz(8.527282248139354) q[16];
sx q[16];
rz(14.60742183593252) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(2.476088295582286) q[4];
cx q[21],q[4];
rz(-2.476088295582286) q[4];
cx q[21],q[4];
rz(3.7514460950218242) q[21];
cx q[21],q[5];
rz(pi/2) q[4];
rz(0.937893768264469) q[4];
sx q[4];
rz(9.170630525011536) q[4];
sx q[4];
rz(8.48688419250491) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(7.891145888544202) q[4];
sx q[4];
rz(5*pi/2) q[4];
rz(pi/4) q[4];
rz(0.8355580949880612) q[4];
rz(-3.7514460950218242) q[5];
sx q[5];
rz(1.6034034901443057) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[21],q[5];
rz(5.544103456687468) q[21];
sx q[21];
rz(4.158784449751313) q[21];
sx q[21];
rz(12.23174295977649) q[21];
rz(0) q[5];
sx q[5];
rz(4.6797818170352805) q[5];
sx q[5];
rz(15.388364805799547) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[7],q[5];
rz(-pi/4) q[5];
cx q[7],q[5];
rz(pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/4) q[5];
rz(pi/4) q[5];
cx q[5],q[3];
rz(-pi/4) q[3];
cx q[5],q[3];
rz(pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(0) q[7];
sx q[7];
rz(6.871432676430989) q[7];
sx q[7];
rz(3*pi) q[7];
rz(1.5392205474021712) q[8];
id q[8];
sx q[8];
cx q[12],q[8];
cx q[8],q[12];
cx q[12],q[8];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[19];
rz(0) q[12];
sx q[12];
rz(3.2228631868179556) q[12];
sx q[12];
rz(3*pi) q[12];
rz(0) q[19];
sx q[19];
rz(3.0603221203616306) q[19];
sx q[19];
rz(3*pi) q[19];
cx q[12],q[19];
rz(-pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
cx q[12],q[17];
rz(0.36394327459081904) q[17];
cx q[12],q[17];
cx q[17],q[6];
rz(-pi/2) q[19];
rz(0.6610853855128112) q[19];
rz(1.4559791806115683) q[6];
cx q[17],q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(2.495176844807014) q[6];
sx q[8];
rz(pi/4) q[8];
cx q[8],q[11];
rz(-pi/4) q[11];
cx q[8],q[11];
rz(pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[11];
cx q[11],q[16];
rz(-pi/4) q[16];
cx q[11],q[16];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[20];
rz(1.7782162684852607) q[20];
cx q[16],q[20];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(1.9650696894042303) q[20];
cx q[3],q[11];
rz(5.172831882468567) q[11];
cx q[3],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/2) q[11];
rz(pi) q[3];
x q[3];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(0.06870713868092393) q[9];
sx q[9];
rz(3.929286197394083) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
cx q[9],q[14];
rz(pi/2) q[14];
rz(2.6894910343205334) q[14];
sx q[14];
rz(5.1554837933329285) q[14];
sx q[14];
rz(11.728498007371027) q[14];
rz(0) q[14];
sx q[14];
rz(4.925742829216385) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[14],q[2];
rz(2.251603218858591) q[2];
cx q[14],q[2];
rz(4.344136682522295) q[14];
rz(0) q[14];
sx q[14];
rz(5.298102429782395) q[14];
sx q[14];
rz(3*pi) q[14];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[7],q[14];
rz(0) q[14];
sx q[14];
rz(0.9850828773971916) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[7],q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
cx q[20],q[14];
rz(-1.9650696894042303) q[14];
cx q[20],q[14];
rz(1.9650696894042303) q[14];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi/4) q[9];
cx q[18],q[9];
rz(-0.6730682232720236) q[9];
cx q[18],q[9];
rz(3.569690916755565) q[18];
cx q[18],q[19];
rz(0.6458077368931402) q[19];
cx q[19],q[15];
rz(-0.6458077368931402) q[15];
cx q[19],q[15];
rz(0.6458077368931402) q[15];
cx q[15],q[6];
cx q[19],q[1];
rz(4.484863371884383) q[1];
cx q[19],q[1];
rz(0.47383190869832253) q[1];
rz(3.3780667096270633) q[1];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(-2.495176844807014) q[6];
cx q[15],q[6];
rz(5.907788728272754) q[15];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/4) q[6];
cx q[16],q[6];
rz(-pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(-pi/2) q[6];
cx q[8],q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[18],q[7];
rz(2.6356993096142287) q[7];
cx q[18],q[7];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-2.218289411117773) q[7];
rz(pi/2) q[7];
rz(pi/4) q[8];
cx q[8],q[5];
rz(-pi/4) q[5];
cx q[8],q[5];
rz(pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(1.3148561671061618) q[8];
cx q[1],q[8];
rz(-3.3780667096270633) q[8];
sx q[8];
rz(1.880687322308611) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[1],q[8];
rz(0) q[8];
sx q[8];
rz(4.402497984870975) q[8];
sx q[8];
rz(11.48798850329028) q[8];
rz(0.6730682232720236) q[9];
rz(-pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[13];
rz(0) q[13];
sx q[13];
rz(1.603779772569044) q[13];
sx q[13];
rz(3*pi) q[13];
rz(0) q[9];
sx q[9];
rz(1.603779772569044) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[9],q[13];
rz(-pi/2) q[13];
rz(-0.4832959992261902) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/4) q[13];
cx q[21],q[13];
cx q[12],q[21];
rz(-pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-pi/2) q[13];
rz(2.4037239136826303) q[13];
rz(3.185549480805635) q[13];
cx q[21],q[12];
cx q[12],q[21];
rz(4.024249416783632) q[21];
rz(pi/2) q[21];
cx q[0],q[21];
rz(0) q[0];
sx q[0];
rz(2.738433438195841) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[21];
sx q[21];
rz(2.738433438195841) q[21];
sx q[21];
rz(3*pi) q[21];
cx q[0],q[21];
rz(-pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[7];
rz(0) q[0];
sx q[0];
rz(3.99135795962301) q[0];
sx q[0];
rz(3*pi) q[0];
rz(-pi/2) q[21];
rz(-4.024249416783632) q[21];
cx q[21],q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/4) q[21];
cx q[21],q[19];
rz(-pi/4) q[19];
cx q[21],q[19];
rz(pi/4) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(0) q[7];
sx q[7];
rz(2.291827347556576) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[0],q[7];
rz(-pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(-pi/2) q[7];
rz(2.218289411117773) q[7];
rz(-pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/4) q[9];
cx q[9],q[2];
rz(-pi/4) q[2];
cx q[9],q[2];
rz(pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-1.9489394501017094) q[2];
cx q[13],q[2];
rz(-3.185549480805635) q[2];
sx q[2];
rz(0.8927477889237978) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[13],q[2];
cx q[17],q[13];
rz(pi/4) q[13];
rz(2.4096740850068645) q[17];
rz(0) q[2];
sx q[2];
rz(5.390437518255789) q[2];
sx q[2];
rz(14.559266891676724) q[2];
sx q[2];
rz(1.2041557554053774) q[2];
sx q[2];
rz(8.251655409922261) q[2];
sx q[2];
rz(12.9431460693716) q[2];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[10],q[9];
rz(pi/4) q[10];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[10],q[9];
rz(-pi/4) q[9];
cx q[10],q[9];
rz(0) q[10];
sx q[10];
rz(5.563959228726331) q[10];
sx q[10];
rz(3*pi) q[10];
cx q[17],q[10];
rz(-2.4096740850068645) q[10];
cx q[17],q[10];
rz(2.4096740850068645) q[10];
rz(pi/4) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[12];
rz(pi/4) q[12];
cx q[9],q[5];
cx q[5],q[9];
cx q[9],q[5];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
measure q[8] -> c[8];
measure q[9] -> c[9];
measure q[10] -> c[10];
measure q[11] -> c[11];
measure q[12] -> c[12];
measure q[13] -> c[13];
measure q[14] -> c[14];
measure q[15] -> c[15];
measure q[16] -> c[16];
measure q[17] -> c[17];
measure q[18] -> c[18];
measure q[19] -> c[19];
measure q[20] -> c[20];
measure q[21] -> c[21];