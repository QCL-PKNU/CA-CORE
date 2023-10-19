OPENQASM 2.0;
include "qelib1.inc";
qreg q[24];
creg c[24];
id q[0];
rz(-pi/2) q[2];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[4],q[5];
rz(-pi/2) q[4];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
x q[5];
rz(-pi/2) q[7];
rz(pi/2) q[9];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[6],q[11];
rz(1.7176016852903464) q[11];
cx q[6],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[12],q[7];
rz(0.2828229777173852) q[12];
rz(pi/2) q[7];
rz(1.5034246817851606) q[7];
rz(pi/2) q[7];
rz(pi) q[13];
x q[13];
cx q[12],q[13];
rz(-0.2828229777173852) q[13];
cx q[12],q[13];
rz(0.2828229777173852) q[13];
rz(5.944770294490878) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[15],q[1];
cx q[1],q[15];
cx q[15],q[1];
rz(3.373169071369543) q[1];
rz(-0.642548487950757) q[1];
cx q[15],q[6];
rz(1.6550336925113776) q[6];
cx q[15],q[6];
rz(0) q[15];
sx q[15];
rz(4.839381936537068) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[5],q[15];
rz(0) q[15];
sx q[15];
rz(1.443803370642518) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[5],q[15];
rz(0.8681618089878266) q[15];
rz(1.674486572794209) q[15];
sx q[15];
rz(8.69187707308447) q[15];
sx q[15];
rz(11.242792723501672) q[15];
rz(4.215963858316257) q[15];
sx q[15];
rz(7.028855993930207) q[15];
sx q[15];
rz(12.27362071685058) q[15];
rz(pi/2) q[15];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(2.6219402162548855) q[6];
cx q[3],q[16];
cx q[16],q[3];
cx q[11],q[3];
rz(4.150211921568633) q[16];
sx q[16];
rz(2.5292606408436353) q[16];
rz(-pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(1.005637019167284) q[3];
cx q[11],q[3];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(3.025053259756538) q[3];
rz(3.932111648942188) q[3];
cx q[3],q[1];
rz(-3.932111648942188) q[1];
sx q[1];
rz(2.576893757084394) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[3],q[1];
rz(0) q[1];
sx q[1];
rz(3.7062915500951923) q[1];
sx q[1];
rz(13.999438097662324) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(-pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[17],q[7];
rz(0) q[17];
sx q[17];
rz(2.0221922799028795) q[17];
sx q[17];
rz(3*pi) q[17];
rz(0) q[7];
sx q[7];
rz(2.0221922799028795) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[17],q[7];
rz(-pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(pi) q[17];
x q[17];
rz(pi/4) q[17];
cx q[17],q[13];
rz(-pi/4) q[13];
cx q[17],q[13];
rz(pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(2.10131236787269) q[13];
sx q[13];
rz(2.2767344305088515) q[13];
rz(-1.307868784278843) q[13];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(-pi/2) q[7];
rz(-1.5034246817851606) q[7];
rz(4.455383129305287) q[7];
sx q[7];
rz(3.694366037307082) q[7];
sx q[7];
rz(14.297409705455893) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(2.161790338207896) q[7];
cx q[18],q[2];
rz(1.3906936792426756) q[18];
cx q[18],q[10];
rz(-1.3906936792426756) q[10];
cx q[18],q[10];
rz(1.3906936792426756) q[10];
rz(1.952202382916902) q[10];
rz(pi/2) q[10];
cx q[16],q[10];
rz(0) q[10];
sx q[10];
rz(0.37090158207164103) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[16];
sx q[16];
rz(0.37090158207164103) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[16],q[10];
rz(-pi/2) q[10];
rz(-1.952202382916902) q[10];
cx q[10],q[5];
rz(-pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[12],q[18];
rz(-pi/4) q[12];
rz(2.4916085790272406) q[12];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(2.669939355806329) q[18];
rz(pi/2) q[18];
rz(pi/2) q[2];
rz(0.6504697824999053) q[2];
rz(3.44474903569248) q[2];
rz(0.20741794579764852) q[5];
cx q[10],q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(0.2839710281854082) q[5];
rz(0.33026297359185774) q[20];
cx q[20],q[14];
rz(-0.33026297359185774) q[14];
cx q[20],q[14];
rz(0.33026297359185774) q[14];
rz(1.2127588352955725) q[14];
cx q[2],q[14];
rz(-3.44474903569248) q[14];
sx q[14];
rz(1.696175108516139) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[2],q[14];
rz(0) q[14];
sx q[14];
rz(4.587010198663448) q[14];
sx q[14];
rz(11.656768161166287) q[14];
rz(4.458816712562433) q[14];
sx q[14];
rz(5.007285239472985) q[14];
rz(-pi/2) q[14];
rz(5.458721753938264) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
x q[20];
rz(-pi/2) q[20];
cx q[20],q[7];
cx q[3],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/4) q[3];
cx q[3],q[2];
rz(-pi/4) q[2];
cx q[3],q[2];
rz(pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-pi/2) q[3];
rz(-2.161790338207896) q[7];
cx q[20],q[7];
cx q[5],q[20];
rz(-0.2839710281854082) q[20];
cx q[5],q[20];
rz(0.2839710281854082) q[20];
rz(pi/2) q[20];
sx q[20];
rz(3.9886322922026407) q[20];
sx q[20];
rz(5*pi/2) q[20];
rz(1.9778875490357373) q[20];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(4.175040353003329) q[5];
cx q[5],q[20];
rz(-4.175040353003329) q[20];
sx q[20];
rz(1.2077860977949302) q[20];
sx q[20];
rz(3*pi) q[20];
cx q[5],q[20];
rz(0) q[20];
sx q[20];
rz(5.075399209384656) q[20];
sx q[20];
rz(11.621930764736971) q[20];
rz(pi/4) q[20];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[20],q[5];
rz(-pi/4) q[5];
cx q[20],q[5];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
cx q[20],q[5];
cx q[5],q[20];
rz(1.7951911280679056) q[20];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-4.783195477211784) q[5];
rz(pi/2) q[5];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(2.5021621564244043) q[7];
sx q[7];
rz(5.563272445025392) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[21],q[19];
cx q[19],q[21];
cx q[21],q[19];
rz(pi/4) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
id q[21];
cx q[4],q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
sx q[19];
rz(0.6847471441059936) q[19];
cx q[19],q[10];
rz(-0.6847471441059936) q[10];
cx q[19],q[10];
rz(0.6847471441059936) q[10];
rz(0.9404068617289191) q[10];
sx q[10];
rz(6.700475692906345) q[10];
sx q[10];
rz(13.556420670769011) q[10];
rz(1.5235752286147288) q[10];
rz(pi/2) q[10];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
cx q[12],q[4];
rz(-2.4916085790272406) q[4];
cx q[12],q[4];
rz(pi/2) q[12];
rz(2.4916085790272406) q[4];
rz(pi) q[4];
x q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[6],q[21];
rz(-2.6219402162548855) q[21];
cx q[6],q[21];
cx q[16],q[6];
rz(2.6219402162548855) q[21];
cx q[21],q[14];
rz(pi/2) q[14];
cx q[14],q[17];
rz(3.031691513449109) q[17];
cx q[14],q[17];
rz(1.7843826974384298) q[14];
rz(2.4323706094048596) q[14];
cx q[14],q[13];
rz(-2.4323706094048596) q[13];
sx q[13];
rz(2.0760195664423273) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[14],q[13];
cx q[12],q[14];
rz(0) q[13];
sx q[13];
rz(4.207165740737259) q[13];
sx q[13];
rz(13.165017354453083) q[13];
rz(3.8227295367191387) q[13];
rz(-pi/2) q[13];
cx q[14],q[12];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(-0.7629692668737237) q[17];
sx q[17];
rz(3.4830973944513564) q[17];
rz(-pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[21];
rz(pi/4) q[21];
cx q[21],q[19];
rz(-pi/4) q[19];
cx q[21],q[19];
rz(pi/4) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(0) q[19];
sx q[19];
rz(5.320503319749463) q[19];
sx q[19];
rz(3*pi) q[19];
rz(-pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[6],q[16];
rz(1.9465191597533977) q[16];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[7],q[4];
rz(5.555200564382013) q[4];
cx q[7],q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-2.8375967632978707) q[4];
rz(pi/2) q[4];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(0.9075315263781226) q[22];
cx q[8],q[22];
rz(-0.9075315263781226) q[22];
cx q[8],q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(-pi/2) q[8];
cx q[0],q[8];
rz(pi/2) q[0];
cx q[11],q[0];
cx q[0],q[11];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[18];
rz(0) q[0];
sx q[0];
rz(0.42493798361958746) q[0];
sx q[0];
rz(3*pi) q[0];
rz(pi/2) q[11];
sx q[11];
rz(3.366557031918828) q[11];
sx q[11];
rz(5*pi/2) q[11];
rz(pi) q[11];
x q[11];
rz(0) q[11];
sx q[11];
rz(5.167787465413244) q[11];
sx q[11];
rz(3*pi) q[11];
rz(0) q[18];
sx q[18];
rz(0.42493798361958746) q[18];
sx q[18];
rz(3*pi) q[18];
cx q[0],q[18];
rz(-pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(-pi/2) q[0];
rz(pi/2) q[0];
rz(-pi/2) q[18];
rz(-2.669939355806329) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[6],q[18];
rz(4.863990761167115) q[18];
cx q[6],q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(2.4022270462583983) q[18];
rz(0.5808932312458251) q[18];
sx q[18];
rz(8.21310088201995) q[18];
sx q[18];
rz(15.664527531687877) q[18];
rz(pi/2) q[18];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(3.5194748607202335) q[6];
rz(-pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[6],q[10];
rz(0) q[10];
sx q[10];
rz(0.8880399471360212) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[6];
sx q[6];
rz(0.8880399471360212) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[6],q[10];
rz(-pi/2) q[10];
rz(-1.5235752286147288) q[10];
rz(-pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
cx q[6],q[14];
cx q[14],q[6];
cx q[6],q[14];
rz(0.8118397313974431) q[14];
rz(pi/2) q[8];
rz(-pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[23],q[9];
cx q[9],q[23];
rz(pi/4) q[23];
rz(-pi/2) q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[8],q[23];
rz(5.1328372308941095) q[23];
cx q[8],q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(pi/2) q[23];
rz(pi/4) q[23];
cx q[23],q[1];
rz(-pi/4) q[1];
cx q[23],q[1];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(2.760692285805784) q[1];
sx q[1];
rz(4.924361140571477) q[1];
sx q[1];
cx q[0],q[1];
x q[0];
rz(-6.14861826334933) q[0];
rz(pi/2) q[0];
rz(0) q[1];
sx q[1];
rz(7.171950486025232) q[1];
sx q[1];
rz(3*pi) q[1];
rz(-pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[1],q[4];
rz(0) q[1];
sx q[1];
rz(5.557447369890539) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[17],q[0];
rz(0) q[0];
sx q[0];
rz(1.4204493344636955) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[17];
sx q[17];
rz(4.862735972715891) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[17],q[0];
rz(-pi/2) q[0];
rz(6.14861826334933) q[0];
rz(-pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
cx q[19],q[17];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(-4.400854987409964) q[23];
rz(pi/2) q[23];
cx q[2],q[23];
rz(0) q[2];
sx q[2];
rz(5.216027261839287) q[2];
sx q[2];
rz(3*pi) q[2];
rz(0) q[23];
sx q[23];
rz(1.0671580453402996) q[23];
sx q[23];
rz(3*pi) q[23];
cx q[2],q[23];
rz(-pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-pi/2) q[23];
rz(4.400854987409964) q[23];
x q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(0) q[4];
sx q[4];
rz(0.7257379372890469) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[1],q[4];
rz(-pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[4];
rz(2.8375967632978707) q[4];
rz(pi) q[4];
x q[4];
rz(pi/4) q[4];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
cx q[8],q[3];
rz(pi/2) q[3];
rz(pi) q[3];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[21];
rz(4.102434128291524) q[21];
cx q[3],q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
cx q[8],q[11];
rz(0) q[11];
sx q[11];
rz(1.1153978417663426) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[8],q[11];
rz(pi/2) q[11];
rz(-0.6975964749427481) q[11];
rz(pi/2) q[8];
sx q[8];
rz(5.4845353725515835) q[8];
sx q[8];
rz(5*pi/2) q[8];
cx q[7],q[8];
rz(4.206917740060072) q[8];
cx q[7],q[8];
rz(pi/4) q[8];
cx q[8],q[1];
rz(-pi/4) q[1];
cx q[8],q[1];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[1];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[22];
rz(1.2361321648447028) q[22];
cx q[9],q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
rz(pi) q[22];
cx q[22],q[16];
rz(-1.9465191597533977) q[16];
cx q[22],q[16];
rz(pi/2) q[16];
rz(pi/2) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[22],q[16];
cx q[16],q[22];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi) q[16];
rz(1.920966970470801) q[16];
cx q[16],q[11];
rz(-1.920966970470801) q[11];
sx q[11];
rz(1.699999528640498) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[16],q[11];
rz(0) q[11];
sx q[11];
rz(4.583185778539088) q[11];
sx q[11];
rz(12.043341406182929) q[11];
cx q[11],q[18];
rz(pi) q[16];
x q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[18],q[11];
cx q[11],q[18];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(1.12234412018738) q[18];
rz(0) q[22];
sx q[22];
rz(8.934678610031018) q[22];
sx q[22];
rz(3*pi) q[22];
cx q[22],q[10];
cx q[10],q[22];
rz(1.7506705587017577) q[10];
cx q[10],q[17];
rz(-1.7506705587017577) q[17];
cx q[10],q[17];
rz(-pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(1.7506705587017577) q[17];
cx q[22],q[3];
rz(4.221822667613485) q[3];
cx q[22],q[3];
sx q[22];
rz(-pi/2) q[22];
cx q[4],q[16];
rz(-pi/4) q[16];
cx q[4],q[16];
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
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
sx q[4];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(-pi/2) q[9];
sx q[9];
rz(pi/4) q[9];
cx q[9],q[2];
rz(-pi/4) q[2];
cx q[9],q[2];
rz(pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[2],q[23];
id q[2];
rz(pi/4) q[2];
cx q[2],q[19];
rz(-pi/4) q[19];
cx q[2],q[19];
rz(pi/4) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
rz(1.6758129387180585) q[19];
rz(-pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[2],q[5];
rz(0) q[2];
sx q[2];
rz(6.228071722732876) q[2];
sx q[2];
rz(3*pi) q[2];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[23],q[12];
rz(4.206224877811172) q[12];
cx q[23],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-2.331492206832936) q[23];
sx q[23];
rz(9.392671867436864) q[23];
sx q[23];
rz(11.756270167602315) q[23];
cx q[20],q[23];
rz(-1.7951911280679056) q[23];
cx q[20],q[23];
rz(1.7951911280679056) q[23];
rz(0) q[5];
sx q[5];
rz(0.05511358444671055) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[2],q[5];
rz(-pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(1.9131124294052713) q[2];
sx q[2];
rz(5.422055445818232) q[2];
rz(-pi/2) q[5];
rz(4.783195477211784) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[7],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
x q[12];
id q[12];
rz(0.2078497368795904) q[12];
rz(-pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[10];
rz(4.138707940493952) q[10];
cx q[7],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/4) q[10];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
sx q[9];
cx q[15],q[9];
x q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[21],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/4) q[21];
cx q[21],q[15];
rz(-pi/4) q[15];
cx q[21],q[15];
rz(pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(-0.2948490289260002) q[21];
sx q[21];
rz(6.869835663219723) q[21];
cx q[21],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[21];
cx q[21],q[11];
rz(-pi/4) q[11];
cx q[21],q[11];
rz(pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(0.6799224156029597) q[11];
cx q[20],q[11];
rz(-0.6799224156029597) q[11];
cx q[20],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(1.7530503639929105) q[11];
rz(3.67068780801078) q[21];
sx q[21];
rz(6.890565922898704) q[21];
sx q[21];
rz(12.322733170025254) q[21];
rz(2.536165551801094) q[21];
cx q[4],q[11];
rz(-1.7530503639929105) q[11];
cx q[4],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[6],q[15];
cx q[15],q[6];
rz(2.67725882400934) q[15];
cx q[15],q[17];
rz(-2.67725882400934) q[17];
cx q[15],q[17];
rz(2.67725882400934) q[17];
rz(3.5201677466122376) q[17];
cx q[17],q[19];
rz(-3.5201677466122376) q[19];
sx q[19];
rz(2.6841169966069027) q[19];
sx q[19];
rz(3*pi) q[19];
cx q[17],q[19];
cx q[17],q[12];
rz(-0.2078497368795904) q[12];
cx q[17],q[12];
rz(5.179155019937628) q[12];
rz(pi/2) q[12];
rz(0) q[19];
sx q[19];
rz(3.5990683105726835) q[19];
sx q[19];
rz(11.269132768663559) q[19];
rz(3.780951437123793) q[19];
sx q[19];
rz(5.961201336052444) q[19];
rz(0.7589057595496538) q[6];
rz(2.212008094796602) q[6];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[0],q[9];
cx q[0],q[13];
cx q[13],q[0];
cx q[0],q[13];
cx q[0],q[1];
cx q[0],q[5];
rz(pi/4) q[0];
rz(-pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[1];
rz(-1.4851218247886027) q[13];
cx q[18],q[1];
cx q[1],q[18];
cx q[18],q[1];
rz(0) q[1];
sx q[1];
rz(4.176241891467605) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[18],q[16];
rz(4.829521651275055) q[16];
cx q[18],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(0.4382884673043652) q[16];
rz(4.047778471790092) q[16];
cx q[18],q[19];
cx q[19],q[18];
cx q[18],q[19];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/4) q[19];
cx q[19],q[18];
rz(-pi/4) q[18];
cx q[19],q[18];
rz(pi/4) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
x q[18];
rz(pi/4) q[18];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[0],q[5];
rz(-pi/4) q[5];
cx q[0],q[5];
rz(-pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[6],q[13];
rz(-2.212008094796602) q[13];
sx q[13];
rz(2.1073165870873387) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[6],q[13];
rz(0) q[13];
sx q[13];
rz(4.1758687200922475) q[13];
sx q[13];
rz(13.121907880354584) q[13];
rz(-pi/2) q[13];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[15],q[6];
rz(pi/4) q[15];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[15],q[6];
rz(-pi/4) q[6];
cx q[15],q[6];
rz(pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[6],q[22];
rz(5.558233380513608) q[22];
cx q[6],q[22];
rz(2.702958582678343) q[22];
rz(0.8646486363660905) q[22];
rz(-4.889708268515261) q[6];
rz(pi/2) q[6];
cx q[4],q[6];
rz(0) q[4];
sx q[4];
rz(3.722299325810366) q[4];
sx q[4];
rz(3*pi) q[4];
rz(0) q[6];
sx q[6];
rz(2.5608859813692204) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[4],q[6];
rz(-pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
cx q[22],q[4];
rz(-0.8646486363660905) q[4];
cx q[22],q[4];
rz(-pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(0.8646486363660905) q[4];
rz(-pi/2) q[6];
rz(4.889708268515261) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
cx q[7],q[1];
rz(0) q[1];
sx q[1];
rz(2.106943415711981) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[7],q[1];
rz(0) q[1];
sx q[1];
rz(4.3570394965772445) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[17],q[7];
rz(0.021380669150179) q[17];
rz(pi/2) q[7];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[14];
rz(-0.8118397313974431) q[14];
cx q[9],q[14];
cx q[14],q[3];
rz(2.109481639429177) q[3];
cx q[14],q[3];
cx q[14],q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(2.50483549146236) q[13];
cx q[14],q[15];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/4) q[14];
rz(-pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[12];
rz(0) q[12];
sx q[12];
rz(1.5641499726109793) q[12];
sx q[12];
rz(3*pi) q[12];
rz(0) q[15];
sx q[15];
rz(1.5641499726109793) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[15],q[12];
rz(-pi/2) q[12];
rz(-5.179155019937628) q[12];
rz(-pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
cx q[2],q[13];
rz(-2.50483549146236) q[13];
cx q[2],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(0.053877026888230914) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(2.1289897280554033) q[13];
cx q[12],q[13];
rz(-2.1289897280554033) q[13];
cx q[12],q[13];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[2],q[1];
rz(0) q[1];
sx q[1];
rz(1.926145810602342) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[2],q[1];
cx q[1],q[14];
rz(-0.1339571916529394) q[1];
rz(-pi/4) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(0.34597142822131816) q[2];
sx q[2];
rz(6.297250615420843) q[2];
sx q[2];
rz(14.437707713923158) q[2];
rz(pi/4) q[2];
cx q[2],q[13];
rz(-pi/4) q[13];
cx q[2],q[13];
rz(pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(3.9198414335170844) q[13];
rz(pi/4) q[2];
cx q[2],q[14];
rz(-pi/4) q[14];
cx q[2],q[14];
rz(pi/4) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/2) q[14];
rz(-1.8361563614508571) q[2];
sx q[2];
rz(7.143730602525166) q[2];
sx q[2];
rz(11.260934322220237) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[23],q[3];
rz(3.1712046101268503) q[3];
cx q[23],q[3];
cx q[23],q[21];
rz(-2.536165551801094) q[21];
cx q[23],q[21];
rz(pi/4) q[21];
rz(5.93456860105938) q[21];
rz(5.73236400129743) q[21];
cx q[21],q[17];
rz(-5.73236400129743) q[17];
sx q[17];
rz(0.15846790383145448) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[21],q[17];
rz(0) q[17];
sx q[17];
rz(6.124717403348132) q[17];
sx q[17];
rz(15.13576129291663) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[12],q[21];
rz(3.5024249394797753) q[21];
cx q[12],q[21];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-0.6780409955018589) q[12];
cx q[13],q[12];
rz(-3.9198414335170844) q[12];
sx q[12];
rz(1.0072293764295033) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[13],q[12];
rz(0) q[12];
sx q[12];
rz(5.275955930750083) q[12];
sx q[12];
rz(14.022660389788323) q[12];
sx q[12];
rz(-pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(1.2137003502387034) q[21];
rz(0.4188174520745129) q[23];
cx q[23],q[11];
rz(-0.4188174520745129) q[11];
cx q[23],q[11];
rz(0.4188174520745129) q[11];
rz(-0.5389970459652329) q[11];
rz(pi/2) q[11];
rz(1.3770601733941397) q[23];
cx q[16],q[23];
rz(-4.047778471790092) q[23];
sx q[23];
rz(1.9530192568973341) q[23];
sx q[23];
rz(3*pi) q[23];
cx q[16],q[23];
rz(0) q[16];
sx q[16];
rz(8.537217124079138) q[16];
sx q[16];
rz(3*pi) q[16];
rz(0) q[23];
sx q[23];
rz(4.330166050282252) q[23];
sx q[23];
rz(12.095496259165332) q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/4) q[3];
cx q[20],q[3];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(-pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[3];
rz(3.4952058030746223) q[3];
sx q[3];
rz(8.242812417188764) q[3];
sx q[3];
rz(15.625340715459664) q[3];
rz(pi/4) q[3];
cx q[3],q[0];
rz(-pi/4) q[0];
cx q[3],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi) q[0];
x q[0];
rz(pi/2) q[0];
cx q[3],q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/4) q[3];
cx q[3],q[19];
rz(-pi/4) q[19];
cx q[3],q[19];
rz(pi/4) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
id q[3];
cx q[3],q[21];
rz(4.621411517698402) q[21];
rz(-pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[4],q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(pi/4) q[4];
cx q[4],q[23];
rz(-pi/4) q[23];
cx q[4],q[23];
rz(pi/4) q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[4],q[0];
cx q[0],q[4];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
cx q[0],q[12];
x q[0];
rz(-pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-0.8485801381973735) q[4];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/4) q[9];
cx q[8],q[9];
rz(0) q[8];
sx q[8];
rz(7.831404801097301) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[8],q[10];
cx q[10],q[8];
cx q[8],q[10];
cx q[10],q[8];
rz(pi/2) q[10];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[10];
cx q[10],q[8];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[8],q[10];
cx q[10],q[8];
cx q[8],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[18],q[10];
rz(-pi/4) q[10];
cx q[18],q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(2.236465019696004) q[10];
cx q[10],q[4];
rz(-2.236465019696004) q[4];
sx q[4];
rz(2.3698766224782655) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[10],q[4];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(0) q[4];
sx q[4];
rz(3.9133086847013208) q[4];
sx q[4];
rz(12.509823118662757) q[4];
rz(pi/2) q[4];
rz(-pi/4) q[8];
rz(-pi/4) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(-pi/2) q[9];
rz(-pi/2) q[9];
rz(2.73344528824424) q[9];
cx q[5],q[9];
rz(-2.73344528824424) q[9];
cx q[5],q[9];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[20],q[5];
rz(2.366638228005098) q[5];
cx q[20],q[5];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(-pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[20],q[11];
rz(0) q[11];
sx q[11];
rz(1.0026860786336815) q[11];
sx q[11];
rz(3*pi) q[11];
rz(0) q[20];
sx q[20];
rz(5.280499228545905) q[20];
sx q[20];
rz(3*pi) q[20];
cx q[20],q[11];
rz(-pi/2) q[11];
rz(0.5389970459652329) q[11];
rz(5.69367900974717) q[11];
cx q[11],q[1];
rz(-5.69367900974717) q[1];
sx q[1];
rz(1.5237375265053004) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[11],q[1];
rz(0) q[1];
sx q[1];
rz(4.759447780674286) q[1];
sx q[1];
rz(15.252414162169488) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[19],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[19],q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(1.1661417944768486) q[14];
rz(-pi/2) q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
rz(pi) q[20];
x q[20];
cx q[23],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[23];
cx q[23],q[1];
rz(-pi/4) q[1];
cx q[23],q[1];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[13],q[1];
rz(1.4563122349778825) q[1];
cx q[13],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
cx q[1],q[2];
rz(-pi/2) q[1];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(2.7194683117872693) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(3.479746441708151) q[5];
sx q[5];
rz(5*pi/2) q[5];
rz(-pi/2) q[5];
cx q[16],q[5];
rz(pi) q[16];
x q[16];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(2.7625621898042323) q[5];
cx q[23],q[5];
rz(-2.7625621898042323) q[5];
cx q[23],q[5];
rz(-pi/4) q[23];
rz(pi/4) q[23];
rz(-1.2559561496571532) q[23];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(4.6743765606854595) q[5];
rz(pi/2) q[5];
cx q[3],q[5];
rz(0) q[3];
sx q[3];
rz(1.6673700166934935) q[3];
sx q[3];
rz(3*pi) q[3];
rz(0) q[5];
sx q[5];
rz(1.6673700166934935) q[5];
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
rz(0) q[3];
sx q[3];
rz(4.59765095972452) q[3];
sx q[3];
rz(3*pi) q[3];
id q[3];
rz(-pi/2) q[5];
rz(-4.6743765606854595) q[5];
rz(1.5191858623350618) q[5];
rz(0) q[5];
sx q[5];
rz(9.262568945923682) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[6],q[20];
cx q[18],q[6];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[6],q[18];
cx q[18],q[14];
rz(-1.1661417944768486) q[14];
cx q[18],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(0.7519534980816658) q[14];
rz(pi/2) q[14];
x q[18];
rz(0) q[18];
sx q[18];
rz(4.6548807892886455) q[18];
sx q[18];
rz(3*pi) q[18];
cx q[6],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/4) q[6];
cx q[6],q[10];
rz(-pi/4) q[10];
cx q[6],q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
sx q[10];
rz(0) q[10];
sx q[10];
rz(3.9093914973249304) q[10];
sx q[10];
rz(3*pi) q[10];
cx q[8],q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(0.2534172846393643) q[20];
cx q[21],q[20];
rz(-4.621411517698402) q[20];
sx q[20];
rz(1.1185171338467241) q[20];
sx q[20];
rz(3*pi) q[20];
cx q[21],q[20];
rz(0) q[20];
sx q[20];
rz(5.164668173332862) q[20];
sx q[20];
rz(13.792772193828416) q[20];
rz(0) q[21];
sx q[21];
rz(7.8604656052148) q[21];
sx q[21];
rz(3*pi) q[21];
rz(0) q[21];
sx q[21];
rz(4.095412660666087) q[21];
sx q[21];
rz(3*pi) q[21];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[4];
cx q[4],q[8];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(1.2265688453048278) q[8];
cx q[8],q[6];
rz(-1.2265688453048278) q[6];
cx q[8],q[6];
rz(1.2265688453048278) q[6];
rz(3.047464120376761) q[6];
sx q[6];
rz(8.602575852236965) q[6];
sx q[6];
rz(11.668124420408796) q[6];
rz(-pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(3.3093778642225384) q[9];
sx q[9];
rz(7.49119108785436) q[9];
sx q[9];
rz(11.972512539178396) q[9];
cx q[15],q[9];
cx q[9],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[7],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/4) q[7];
cx q[7],q[15];
rz(-pi/4) q[15];
cx q[7],q[15];
rz(pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
sx q[15];
rz(1.1376740537255003) q[15];
cx q[17],q[7];
cx q[7],q[17];
rz(3.71382989037085) q[17];
sx q[17];
rz(4.065187831556565) q[17];
sx q[17];
rz(10.744301100881689) q[17];
rz(-pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[17],q[0];
rz(1.987929879983642) q[0];
cx q[17],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(2.0752989465524863) q[17];
rz(2.467757765629838) q[17];
rz(2.736807116943515) q[17];
cx q[17],q[23];
rz(-2.736807116943515) q[23];
sx q[23];
rz(2.5793475272279682) q[23];
sx q[23];
rz(3*pi) q[23];
cx q[17],q[23];
rz(0) q[23];
sx q[23];
rz(3.703837779951618) q[23];
sx q[23];
rz(13.417541227370048) q[23];
rz(pi/4) q[7];
cx q[7],q[11];
rz(-pi/4) q[11];
cx q[7],q[11];
rz(pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[11],q[12];
rz(3.335043312417748) q[12];
cx q[11],q[12];
cx q[11],q[4];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(0.9541367800102604) q[12];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/4) q[7];
cx q[12],q[7];
rz(-0.9541367800102604) q[7];
cx q[12],q[7];
rz(-pi/2) q[12];
rz(0.9541367800102604) q[7];
rz(-1.4732950534242302) q[9];
rz(pi/2) q[9];
cx q[22],q[9];
rz(0) q[22];
sx q[22];
rz(4.7379519771963725) q[22];
sx q[22];
rz(3*pi) q[22];
rz(0) q[9];
sx q[9];
rz(1.5452333299832142) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[22],q[9];
rz(-pi/2) q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(1.3227121965350217) q[22];
rz(-pi/2) q[9];
rz(1.4732950534242302) q[9];
cx q[9],q[22];
rz(-1.3227121965350217) q[22];
cx q[9],q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(3.872994026660698) q[22];
cx q[22],q[15];
rz(-3.872994026660698) q[15];
sx q[15];
rz(2.7894386532242894) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[22],q[15];
rz(0) q[15];
sx q[15];
rz(3.493746653955297) q[15];
sx q[15];
rz(12.160097933704577) q[15];
rz(-pi/4) q[15];
cx q[15],q[1];
cx q[0],q[15];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[1];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(2.7805478331958833) q[22];
cx q[19],q[22];
rz(-2.7805478331958833) q[22];
cx q[19],q[22];
rz(-pi/2) q[19];
cx q[20],q[19];
rz(pi/2) q[19];
rz(-1.2173668860349316) q[19];
rz(pi/2) q[19];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[22],q[13];
rz(0.8988948631508057) q[13];
cx q[22],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(2.704008600426895) q[13];
sx q[13];
rz(6.8555002875762785) q[13];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[4],q[1];
rz(-pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[1];
cx q[8],q[19];
rz(0) q[19];
sx q[19];
rz(1.5194059311586083) q[19];
sx q[19];
rz(3*pi) q[19];
rz(0) q[8];
sx q[8];
rz(4.7637793760209775) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[8],q[19];
rz(-pi/2) q[19];
rz(1.2173668860349316) q[19];
rz(-pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[9];
cx q[9],q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
cx q[20],q[16];
cx q[16],q[20];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[9],q[2];
cx q[2],q[9];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[7],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/4) q[7];
cx q[7],q[2];
rz(-pi/4) q[2];
cx q[7],q[2];
rz(pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[9],q[22];
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
measure q[22] -> c[22];
measure q[23] -> c[23];