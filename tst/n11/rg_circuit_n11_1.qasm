OPENQASM 2.0;
include "qelib1.inc";
qreg q[11];
creg c[11];
rz(2.0493309632593713) q[0];
rz(4.520523175389697) q[1];
rz(4.1983701481493405) q[1];
cx q[1],q[0];
rz(-4.1983701481493405) q[0];
sx q[0];
rz(2.8971018579202554) q[0];
sx q[0];
rz(3*pi) q[0];
cx q[1],q[0];
rz(0) q[0];
sx q[0];
rz(3.386083449259331) q[0];
sx q[0];
rz(11.573817145659348) q[0];
rz(0.654049244289781) q[0];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-pi/2) q[3];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/4) q[5];
cx q[5],q[1];
rz(-pi/4) q[1];
cx q[5],q[1];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(2.3634828086981394) q[1];
sx q[1];
rz(3.5886596858559097) q[1];
rz(-pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
id q[5];
rz(4.97572112061126) q[5];
sx q[5];
rz(7.669046354685496) q[5];
sx q[5];
rz(10.725026710319947) q[5];
rz(0.971323555898391) q[5];
sx q[6];
rz(2.171562925316847) q[6];
rz(4.859324945843388) q[6];
cx q[6],q[0];
rz(-4.859324945843388) q[0];
sx q[0];
rz(3.1119685385019547) q[0];
sx q[0];
rz(3*pi) q[0];
cx q[6],q[0];
rz(0) q[0];
sx q[0];
rz(3.1712167686776316) q[0];
sx q[0];
rz(13.630053662322986) q[0];
rz(-pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi) q[7];
x q[7];
rz(pi) q[7];
rz(pi/2) q[7];
rz(4.580139131287835) q[8];
rz(-0.33418219437908037) q[8];
sx q[8];
rz(3.646119267640515) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[7];
cx q[7],q[8];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[4],q[9];
rz(-pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[4],q[3];
rz(5.209716911494557) q[3];
cx q[4],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(-pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[4],q[3];
rz(5.176815819380688) q[3];
cx q[4],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(5.149936231191095) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(4.346643974292514) q[4];
sx q[4];
rz(3.6975173863268163) q[4];
rz(6.0124021837992006) q[4];
sx q[4];
rz(4.188527511919231) q[4];
sx q[4];
rz(13.543072379307034) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(0) q[4];
sx q[4];
rz(5.253446264404225) q[4];
sx q[4];
rz(3*pi) q[4];
rz(-pi/2) q[4];
rz(3.549188099643416) q[10];
rz(pi/2) q[10];
cx q[2],q[10];
rz(0) q[10];
sx q[10];
rz(0.7703303875732606) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[2];
sx q[2];
rz(0.7703303875732606) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[2],q[10];
rz(-pi/2) q[10];
rz(-3.549188099643416) q[10];
id q[10];
rz(5.138050784694504) q[10];
rz(pi/2) q[10];
rz(-pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
cx q[6],q[10];
rz(0) q[10];
sx q[10];
rz(2.589458157916834) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[6];
sx q[6];
rz(2.589458157916834) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[6],q[10];
rz(-pi/2) q[10];
rz(-5.138050784694504) q[10];
rz(1.4092623486455516) q[10];
rz(-pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
cx q[6],q[7];
rz(4.062359873824651) q[7];
cx q[6],q[7];
rz(3.09667979542678) q[6];
rz(4.101004654426813) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[7];
sx q[7];
rz(8.984594446357713) q[7];
sx q[7];
rz(5*pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[9],q[2];
rz(0.060020289838281775) q[2];
cx q[9],q[2];
rz(1.8371104052231657) q[2];
cx q[2],q[0];
rz(-1.8371104052231657) q[0];
cx q[2],q[0];
rz(1.8371104052231657) q[0];
rz(-6.266387784511429) q[0];
rz(pi/2) q[0];
cx q[1],q[0];
rz(0) q[0];
sx q[0];
rz(0.9438179458249136) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[1];
sx q[1];
rz(5.339367361354673) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[1],q[0];
rz(-pi/2) q[0];
rz(6.266387784511429) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(0.32358477204769476) q[1];
rz(0) q[2];
sx q[2];
rz(6.221563692592339) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[2],q[10];
cx q[10],q[2];
cx q[2],q[10];
rz(1.2080332665487499) q[10];
rz(pi/4) q[2];
cx q[5],q[1];
rz(-0.971323555898391) q[1];
sx q[1];
rz(0.38642418761193387) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[5],q[1];
rz(0) q[1];
sx q[1];
rz(5.896761119567652) q[1];
sx q[1];
rz(10.072516744620076) q[1];
rz(5.70910245055789) q[1];
sx q[1];
rz(6.802490044308755) q[1];
sx q[1];
rz(11.632470283296803) q[1];
rz(0.7338564976205579) q[1];
rz(pi/4) q[5];
cx q[5],q[7];
rz(-pi/4) q[7];
cx q[5],q[7];
rz(pi/2) q[5];
cx q[6],q[5];
cx q[5],q[6];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
id q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[5];
rz(pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi) q[9];
x q[9];
rz(-pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[8],q[9];
rz(2.429445649207635) q[9];
cx q[8],q[9];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[0],q[8];
rz(2.172344778459351) q[8];
cx q[0],q[8];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[10];
rz(-1.2080332665487499) q[10];
cx q[0],q[10];
rz(2.853688288902724) q[0];
cx q[0],q[1];
rz(-2.853688288902724) q[1];
sx q[1];
rz(0.6144617318432539) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[0],q[1];
cx q[0],q[4];
rz(0) q[1];
sx q[1];
rz(5.668723575336332) q[1];
sx q[1];
rz(11.544609752051546) q[1];
rz(pi/4) q[1];
cx q[1],q[7];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-pi/4) q[7];
cx q[1],q[7];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/4) q[7];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[3],q[9];
rz(pi/4) q[3];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[3],q[9];
rz(-pi/4) q[9];
cx q[3],q[9];
sx q[3];
cx q[8],q[3];
rz(-pi/2) q[3];
sx q[3];
cx q[0],q[3];
cx q[3],q[0];
cx q[0],q[3];
rz(0) q[0];
sx q[0];
rz(7.674426556444668) q[0];
sx q[0];
rz(3*pi) q[0];
rz(1.342418232803222) q[0];
rz(pi/2) q[0];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
x q[8];
rz(0) q[8];
sx q[8];
rz(3.4650363197546694) q[8];
sx q[8];
rz(3*pi) q[8];
rz(pi/4) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[2],q[9];
rz(-pi/4) q[9];
cx q[2],q[9];
rz(pi/2) q[2];
cx q[10],q[2];
cx q[2],q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
cx q[1],q[10];
cx q[10],q[1];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(0) q[10];
sx q[10];
rz(5.778569605296077) q[10];
sx q[10];
rz(3*pi) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/4) q[10];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-1.6070859984846653) q[2];
rz(pi/2) q[2];
rz(pi/4) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[8];
rz(0) q[8];
sx q[8];
rz(2.818148987424917) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[9],q[8];
cx q[6],q[8];
cx q[6],q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[4],q[1];
cx q[1],q[4];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[1],q[4];
rz(2.665914835845941) q[4];
cx q[1],q[4];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[6],q[7];
rz(1.5229483336499954) q[6];
cx q[6],q[5];
rz(pi/2) q[5];
rz(pi/2) q[6];
rz(-pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi/2) q[7];
rz(-pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[0];
rz(0) q[0];
sx q[0];
rz(2.281652654575015) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[7];
sx q[7];
rz(2.281652654575015) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[7],q[0];
rz(-pi/2) q[0];
rz(-1.342418232803222) q[0];
rz(0.36407493231227583) q[0];
rz(4.786199354059898) q[0];
rz(-pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(-pi/4) q[7];
rz(4.022594964093854) q[7];
rz(pi/4) q[7];
rz(-pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[2];
rz(0) q[2];
sx q[2];
rz(2.3192281739957155) q[2];
sx q[2];
rz(3*pi) q[2];
rz(0) q[9];
sx q[9];
rz(3.9639571331838708) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[9],q[2];
rz(-pi/2) q[2];
rz(1.6070859984846653) q[2];
rz(2.717349426784299) q[2];
rz(-pi/4) q[2];
sx q[2];
rz(-1.4446676903537905) q[2];
cx q[0],q[2];
rz(-4.786199354059898) q[2];
sx q[2];
rz(1.1856065647345853) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[0],q[2];
cx q[0],q[5];
rz(0) q[2];
sx q[2];
rz(5.097578742445001) q[2];
sx q[2];
rz(15.655645005183068) q[2];
cx q[5],q[0];
cx q[0],q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
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
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
x q[7];
rz(0.03181134002463354) q[7];
rz(-pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[8],q[9];
rz(5.9010297506145735) q[9];
cx q[8],q[9];
rz(pi) q[8];
rz(1.9127400868664253) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(-pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(-pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[3];
rz(3.4868168196104623) q[3];
cx q[9],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
cx q[3],q[10];
cx q[1],q[3];
rz(-pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[3],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
rz(3.3252579147842365) q[1];
cx q[2],q[0];
cx q[0],q[2];
cx q[2],q[0];
rz(1.837978208971495) q[0];
rz(2.274939979133005) q[2];
sx q[2];
rz(4.767386888448973) q[2];
sx q[2];
rz(9.778469105307861) q[2];
sx q[2];
cx q[3],q[10];
rz(0.052285251361104144) q[10];
cx q[3],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(4.808053017355412) q[10];
rz(pi/2) q[10];
rz(0.29860218535268634) q[3];
cx q[8],q[10];
rz(0) q[10];
sx q[10];
rz(2.751227722526406) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[8];
sx q[8];
rz(2.751227722526406) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[8],q[10];
rz(-pi/2) q[10];
rz(-4.808053017355412) q[10];
cx q[0],q[10];
rz(-1.837978208971495) q[10];
cx q[0],q[10];
rz(0.6255488903233091) q[0];
rz(0.49031223465316914) q[0];
cx q[0],q[7];
rz(1.837978208971495) q[10];
rz(-0.49031223465316914) q[7];
sx q[7];
rz(0.6404990865100921) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[0],q[7];
rz(1.7645244474803077) q[0];
rz(0) q[7];
sx q[7];
rz(5.6426862206694945) q[7];
sx q[7];
rz(9.883278855397915) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(-pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi) q[9];
rz(-3.2753733332793726) q[9];
rz(pi/2) q[9];
cx q[4],q[9];
rz(0) q[4];
sx q[4];
rz(3.3574962789826257) q[4];
sx q[4];
rz(3*pi) q[4];
rz(0) q[9];
sx q[9];
rz(2.9256890281969605) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[4],q[9];
rz(-pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
x q[4];
rz(pi/4) q[4];
rz(4.782951076513135) q[4];
sx q[4];
rz(5.213319538083502) q[4];
sx q[4];
rz(12.095592219871172) q[4];
rz(-pi/2) q[9];
rz(3.2753733332793726) q[9];
sx q[9];
cx q[6],q[9];
x q[6];
rz(5.27950915365269) q[6];
cx q[6],q[3];
rz(-5.27950915365269) q[3];
sx q[3];
rz(2.697660209496619) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[6],q[3];
rz(0) q[3];
sx q[3];
rz(3.585525097682967) q[3];
sx q[3];
rz(14.405684929069384) q[3];
id q[3];
rz(-pi/2) q[6];
rz(-pi/4) q[6];
rz(2.4943012951525816) q[9];
cx q[1],q[9];
rz(-3.3252579147842365) q[9];
sx q[9];
rz(1.655161471486278) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[1],q[9];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[5],q[1];
rz(5.286374890228368) q[1];
cx q[5],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(5.054139350269569) q[1];
cx q[1],q[4];
cx q[4],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[1],q[7];
rz(1.75494057475752) q[4];
rz(pi/2) q[5];
cx q[5],q[2];
rz(-pi/2) q[2];
x q[5];
rz(pi) q[5];
x q[5];
rz(-pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[6],q[2];
rz(pi/2) q[2];
rz(-pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(9.238853559977514) q[2];
sx q[2];
rz(5*pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(1.0029230272519125) q[7];
cx q[1],q[7];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[4],q[1];
rz(0.8152931753404545) q[1];
cx q[4],q[1];
cx q[1],q[2];
rz(1.7401946735742633) q[2];
cx q[1],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(3.5288709422372126) q[4];
sx q[4];
rz(3.8598726058004806) q[4];
sx q[4];
rz(15.568886948940946) q[4];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(0) q[9];
sx q[9];
rz(4.628023835693308) q[9];
sx q[9];
rz(10.255734580401034) q[9];
rz(-5.466076184892195) q[9];
rz(pi/2) q[9];
cx q[8],q[9];
rz(0) q[8];
sx q[8];
rz(5.883774821092786) q[8];
sx q[8];
rz(3*pi) q[8];
rz(0) q[9];
sx q[9];
rz(0.3994104860868011) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[8],q[9];
rz(-pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(2.7973031194822444) q[8];
cx q[3],q[8];
rz(-2.7973031194822444) q[8];
cx q[3],q[8];
rz(2.93824088139237) q[3];
cx q[3],q[0];
rz(-2.93824088139237) q[0];
sx q[0];
rz(2.6547055434267905) q[0];
sx q[0];
rz(3*pi) q[0];
cx q[3],q[0];
rz(0) q[0];
sx q[0];
rz(3.6284797637527957) q[0];
sx q[0];
rz(10.598494394681442) q[0];
cx q[0],q[6];
rz(-pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(2.815293150379986) q[3];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[7],q[0];
rz(0.26719099726999346) q[0];
cx q[7],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi) q[8];
x q[8];
rz(-pi/2) q[9];
rz(5.466076184892195) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[10],q[9];
rz(3.4385169780097264) q[9];
cx q[10],q[9];
rz(pi/2) q[10];
sx q[10];
rz(6.776486990444887) q[10];
sx q[10];
rz(5*pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(0.41729109171327916) q[10];
cx q[8],q[10];
rz(-0.41729109171327916) q[10];
cx q[8],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi) q[10];
x q[10];
cx q[8],q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
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
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[5],q[7];
rz(3.8423795162051757) q[7];
cx q[5],q[7];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[0];
rz(4.065628882424282) q[0];
cx q[8],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi) q[9];
cx q[3],q[9];
rz(-2.815293150379986) q[9];
cx q[3],q[9];
rz(0.20046007448964467) q[3];
cx q[3],q[6];
rz(-0.20046007448964467) q[6];
cx q[3],q[6];
rz(0.20046007448964467) q[6];
rz(3.0564938500770795) q[6];
rz(2.815293150379986) q[9];
rz(pi/4) q[9];
cx q[3],q[9];
cx q[9],q[3];
cx q[3],q[9];
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
