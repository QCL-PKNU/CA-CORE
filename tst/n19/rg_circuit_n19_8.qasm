OPENQASM 2.0;
include "qelib1.inc";
qreg q[19];
creg c[19];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(0.014639204220825168) q[3];
id q[3];
rz(pi/2) q[3];
sx q[3];
rz(3.9573717217123994) q[3];
sx q[3];
rz(5*pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(5.242768040285279) q[3];
sx q[3];
rz(5*pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(-pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-5.365291011956896) q[5];
rz(pi/2) q[5];
cx q[2],q[5];
rz(0) q[2];
sx q[2];
rz(3.270866804327093) q[2];
sx q[2];
rz(3*pi) q[2];
rz(0) q[5];
sx q[5];
rz(3.012318502852493) q[5];
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
rz(-pi/4) q[2];
rz(-pi/2) q[5];
rz(5.365291011956896) q[5];
rz(pi) q[5];
x q[5];
sx q[5];
rz(-1.387674331379047) q[6];
rz(pi/4) q[8];
cx q[8],q[0];
rz(-pi/4) q[0];
cx q[8],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
id q[0];
rz(3.1865643915600534) q[0];
rz(1.8407754416232698) q[0];
rz(2.2608165709385943) q[9];
sx q[9];
rz(8.49035461351502) q[9];
sx q[9];
rz(13.033744895181501) q[9];
rz(pi/4) q[9];
x q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi) q[10];
x q[10];
rz(pi/4) q[10];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(2.860440442077788) q[11];
rz(0.9468042133139516) q[12];
cx q[12],q[7];
rz(-0.9468042133139516) q[7];
cx q[12],q[7];
rz(1.5808281043062167) q[12];
rz(0.9468042133139516) q[7];
rz(pi) q[13];
x q[13];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(2.7141791652795235) q[14];
cx q[14],q[6];
rz(-2.7141791652795235) q[6];
sx q[6];
rz(2.20919312925271) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[14],q[6];
cx q[14],q[8];
rz(0) q[6];
sx q[6];
rz(4.073992177926876) q[6];
sx q[6];
rz(13.52663145742795) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[10],q[6];
rz(-pi/4) q[6];
cx q[10],q[6];
rz(-pi/2) q[10];
rz(pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/4) q[6];
cx q[2],q[6];
rz(1.1009444716196242) q[2];
rz(-pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(-pi/2) q[6];
rz(-4.570225940382967) q[6];
sx q[6];
rz(6.8072180621714775) q[6];
sx q[6];
rz(13.995003901152346) q[6];
rz(0.7093497111446943) q[8];
cx q[14],q[8];
rz(2.799355682970195) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[8];
cx q[8],q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(3.026645724283069) q[5];
x q[8];
cx q[8],q[9];
rz(5.436153209903936) q[9];
cx q[8],q[9];
rz(1.2732537927287528) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(-pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[1];
rz(5.719314555622519) q[1];
cx q[15],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi) q[1];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(3.966615258949582) q[15];
rz(pi/2) q[15];
cx q[13],q[15];
rz(0) q[13];
sx q[13];
rz(1.1168048670341175) q[13];
sx q[13];
rz(3*pi) q[13];
rz(0) q[15];
sx q[15];
rz(1.1168048670341175) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[13],q[15];
rz(-pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
x q[13];
id q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-pi/2) q[15];
rz(-3.966615258949582) q[15];
cx q[15],q[10];
rz(pi/2) q[10];
rz(5.640630177456336) q[10];
rz(2.372035099726121) q[10];
cx q[10],q[2];
cx q[15],q[5];
rz(-2.372035099726121) q[2];
sx q[2];
rz(1.453286170002031) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[10],q[2];
rz(0) q[2];
sx q[2];
rz(4.829899137177556) q[2];
sx q[2];
rz(10.695868588875877) q[2];
rz(-3.026645724283069) q[5];
cx q[15],q[5];
rz(0) q[15];
sx q[15];
rz(5.912971003994054) q[15];
sx q[15];
rz(3*pi) q[15];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(1.6022364296597267) q[5];
sx q[5];
rz(3.703128111677802) q[5];
sx q[5];
rz(14.610420199732246) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[6],q[15];
rz(0) q[15];
sx q[15];
rz(0.37021430318553294) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[6],q[15];
rz(-0.7013262088284058) q[6];
sx q[6];
rz(7.631933295769144) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[8],q[2];
rz(-1.2732537927287528) q[2];
cx q[8],q[2];
rz(1.2732537927287528) q[2];
rz(pi/4) q[2];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[5],q[8];
rz(0.21844539456747367) q[8];
cx q[5],q[8];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
x q[5];
rz(-4.509355464190195) q[5];
sx q[5];
rz(3.574460477390907) q[5];
sx q[5];
rz(13.934133424959574) q[5];
id q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(1.3715383350258457) q[5];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[9],q[13];
rz(3.0871019266811244) q[13];
cx q[9],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[16],q[17];
cx q[12],q[17];
cx q[16],q[11];
rz(-2.860440442077788) q[11];
cx q[16],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
sx q[11];
rz(0) q[16];
sx q[16];
rz(4.485427529351928) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[1],q[16];
rz(0) q[16];
sx q[16];
rz(1.797757777827658) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[1],q[16];
rz(pi/2) q[1];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[14],q[16];
rz(2.1649027385089203) q[16];
cx q[14],q[16];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
x q[14];
rz(-pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-1.5808281043062167) q[17];
cx q[12],q[17];
rz(1.5808281043062167) q[17];
cx q[17],q[12];
cx q[12],q[17];
cx q[17],q[12];
rz(-pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[12],q[17];
rz(0.6039874578705635) q[17];
cx q[12],q[17];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
cx q[12],q[10];
cx q[10],q[12];
rz(pi/2) q[10];
sx q[10];
rz(8.847057721707369) q[10];
sx q[10];
rz(5*pi/2) q[10];
x q[10];
rz(5.325080444676768) q[10];
sx q[10];
rz(4.611187978541978) q[10];
sx q[10];
rz(14.067917938516446) q[10];
rz(pi) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
cx q[17],q[13];
rz(0) q[13];
sx q[13];
rz(3.7182790708593743) q[13];
sx q[13];
rz(3*pi) q[13];
rz(2.051516140014402) q[17];
rz(3.981665376520497) q[17];
sx q[17];
rz(7.609652448803904) q[17];
rz(pi/4) q[17];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/4) q[18];
cx q[4],q[18];
rz(-pi/4) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi/2) q[18];
rz(pi) q[18];
x q[18];
rz(0) q[18];
sx q[18];
rz(5.685239746079055) q[18];
sx q[18];
rz(3*pi) q[18];
rz(2.556756520069762) q[18];
cx q[18],q[0];
rz(-2.556756520069762) q[0];
sx q[0];
rz(1.9119137463566034) q[0];
sx q[0];
rz(3*pi) q[0];
cx q[18],q[0];
rz(0) q[0];
sx q[0];
rz(4.371271560822983) q[0];
sx q[0];
rz(10.14075903921587) q[0];
rz(2.503476933968649) q[0];
sx q[0];
rz(2.463045342324099) q[0];
rz(pi/4) q[0];
rz(-2.2346380920499613) q[0];
rz(pi/2) q[0];
rz(pi/2) q[18];
rz(2.0799222321764743) q[4];
cx q[7],q[4];
rz(-2.0799222321764743) q[4];
cx q[7],q[4];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[4],q[7];
rz(1.0512563257664163) q[4];
cx q[4],q[11];
rz(-1.0512563257664163) q[11];
cx q[4],q[11];
rz(1.0512563257664163) q[11];
rz(pi/2) q[11];
sx q[11];
rz(7.175826981206755) q[11];
sx q[11];
rz(5*pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[2],q[11];
rz(-pi/4) q[11];
cx q[2],q[11];
rz(pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(3.636059761395422) q[11];
rz(pi/2) q[11];
rz(-pi/2) q[2];
cx q[3],q[11];
rz(0) q[11];
sx q[11];
rz(0.4287935864512078) q[11];
sx q[11];
rz(3*pi) q[11];
rz(0) q[3];
sx q[3];
rz(0.4287935864512078) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[3],q[11];
rz(-pi/2) q[11];
rz(-3.636059761395422) q[11];
rz(-pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[1];
cx q[1],q[7];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
cx q[16],q[1];
cx q[1],q[16];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(1.8840416422223174) q[1];
sx q[1];
rz(9.242480226133083) q[1];
sx q[1];
rz(9.90418641944857) q[1];
cx q[1],q[8];
rz(-4.968888556781442) q[16];
rz(pi/2) q[16];
cx q[14],q[16];
rz(0) q[14];
sx q[14];
rz(5.181534015605708) q[14];
sx q[14];
rz(3*pi) q[14];
rz(0) q[16];
sx q[16];
rz(1.1016512915738779) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[14],q[16];
rz(-pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/4) q[14];
cx q[14],q[6];
rz(-pi/2) q[16];
rz(4.968888556781442) q[16];
rz(-pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[11],q[16];
rz(3.521100630817412) q[16];
cx q[11],q[16];
rz(pi/2) q[11];
sx q[11];
rz(5.321800928054559) q[11];
sx q[11];
rz(5*pi/2) q[11];
rz(pi/4) q[11];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-pi/4) q[6];
cx q[14],q[6];
rz(pi/4) q[14];
rz(2.112929733705192) q[14];
rz(pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[12],q[6];
rz(0.76633834961568) q[6];
cx q[12],q[6];
x q[12];
sx q[12];
rz(4.3094614476244955) q[6];
sx q[6];
rz(3.903013117514784) q[6];
sx q[6];
rz(pi/4) q[7];
cx q[7],q[4];
rz(-pi/4) q[4];
cx q[7],q[4];
rz(pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[4],q[15];
rz(5.740432739958255) q[15];
rz(pi) q[15];
rz(2.237112865084465) q[15];
cx q[15],q[14];
rz(-2.237112865084465) q[14];
sx q[14];
rz(2.624913312118248) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[15],q[14];
rz(0) q[14];
sx q[14];
rz(3.6582719950613383) q[14];
sx q[14];
rz(9.548961092148653) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[17],q[14];
rz(5.387833098578721) q[14];
cx q[17],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[4],q[2];
rz(pi/2) q[2];
rz(pi) q[2];
x q[2];
cx q[16],q[2];
cx q[2],q[16];
cx q[16],q[2];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(1.6058013522211696) q[16];
rz(0) q[2];
sx q[2];
rz(3.775489922848774) q[2];
sx q[2];
rz(3*pi) q[2];
rz(pi/2) q[2];
rz(pi) q[4];
x q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(1.8949140670218774) q[4];
rz(-pi/2) q[7];
rz(-pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[0];
rz(0) q[0];
sx q[0];
rz(0.034359343347790894) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[7];
sx q[7];
rz(6.248825963831795) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[7],q[0];
rz(-pi/2) q[0];
rz(2.2346380920499613) q[0];
rz(-pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
cx q[8],q[1];
rz(-pi/2) q[1];
cx q[7],q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[3],q[1];
rz(5.419140380854984) q[1];
cx q[3],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[3];
cx q[3],q[6];
x q[3];
sx q[3];
cx q[2],q[3];
x q[2];
rz(3.298226485896805) q[2];
rz(-0.030394964149858428) q[3];
rz(1.753775068709516) q[6];
sx q[6];
rz(3.52963372270177) q[6];
rz(1.7439917738811672) q[7];
cx q[7],q[15];
rz(5.875900363543352) q[15];
cx q[7],q[15];
rz(-6.198836979551202) q[15];
rz(pi/2) q[15];
sx q[7];
rz(pi) q[8];
rz(0.6650270910757909) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[11],q[8];
rz(-pi/4) q[8];
cx q[11],q[8];
rz(-pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[11],q[15];
rz(0) q[11];
sx q[11];
rz(5.609711930667334) q[11];
sx q[11];
rz(3*pi) q[11];
rz(0) q[15];
sx q[15];
rz(0.673473376512252) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[11],q[15];
rz(-pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(-pi/2) q[15];
rz(6.198836979551202) q[15];
rz(-1.0127771099125937) q[15];
cx q[2],q[15];
rz(-3.298226485896805) q[15];
sx q[15];
rz(3.1019035675631805) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[2],q[15];
rz(0) q[15];
sx q[15];
rz(3.1812817396164057) q[15];
sx q[15];
rz(13.735781556578779) q[15];
rz(pi/4) q[15];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[15],q[2];
rz(-pi/4) q[2];
cx q[15],q[2];
rz(pi/4) q[15];
rz(pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(0.10423452122193654) q[2];
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[9],q[18];
cx q[18],q[9];
rz(1.7707865068530377) q[18];
sx q[18];
rz(4.582205660334401) q[18];
sx q[18];
rz(13.46280296001415) q[18];
rz(pi) q[18];
cx q[18],q[4];
rz(-1.8949140670218774) q[4];
cx q[18],q[4];
rz(pi/2) q[18];
cx q[18],q[12];
rz(pi/2) q[12];
cx q[12],q[7];
x q[12];
rz(5.68329078446855) q[12];
cx q[12],q[3];
x q[18];
rz(0) q[18];
sx q[18];
rz(3.443599992787823) q[18];
sx q[18];
rz(3*pi) q[18];
rz(-5.68329078446855) q[3];
sx q[3];
rz(2.1486597333899713) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[12],q[3];
rz(0.02802737802464387) q[12];
rz(0) q[3];
sx q[3];
rz(4.134525573789615) q[3];
sx q[3];
rz(15.13846370938779) q[3];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[4],q[16];
rz(-1.6058013522211696) q[16];
cx q[4],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-2.9682953971927253) q[16];
rz(pi/2) q[16];
cx q[14],q[16];
rz(0) q[14];
sx q[14];
rz(5.587132947099004) q[14];
sx q[14];
rz(3*pi) q[14];
rz(0) q[16];
sx q[16];
rz(0.6960523600805826) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[14],q[16];
rz(-pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
cx q[14],q[6];
rz(-pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/2) q[16];
rz(2.9682953971927253) q[16];
rz(pi) q[16];
rz(-pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[3],q[14];
rz(0.13438872950294972) q[14];
cx q[3],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
cx q[14],q[2];
rz(-0.10423452122193654) q[2];
cx q[14],q[2];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/4) q[2];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(4.21148064747083) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[2],q[3];
rz(-pi/4) q[3];
cx q[2],q[3];
rz(pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/4) q[3];
rz(pi/4) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(4.641348629689957) q[6];
sx q[6];
rz(4.500361294445739) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[15],q[6];
rz(-pi/4) q[6];
cx q[15],q[6];
rz(pi/2) q[15];
rz(pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(-4.625301825530094) q[6];
sx q[6];
rz(9.086757431485923) q[6];
sx q[6];
rz(14.050079786299474) q[6];
rz(0) q[6];
sx q[6];
rz(3.9187004344073086) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[2],q[6];
rz(0) q[6];
sx q[6];
rz(2.3644848727722776) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[2],q[6];
rz(2.827640433201767) q[2];
sx q[2];
rz(7.373980916937157) q[2];
sx q[2];
rz(10.682103668659305) q[2];
rz(-0.9879834570245444) q[2];
id q[6];
cx q[8],q[18];
rz(0) q[18];
sx q[18];
rz(2.8395853143917633) q[18];
sx q[18];
rz(3*pi) q[18];
cx q[8],q[18];
rz(0) q[18];
sx q[18];
rz(4.571716137694133) q[18];
sx q[18];
rz(3*pi) q[18];
rz(-pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[9],q[13];
rz(0) q[13];
sx q[13];
rz(2.564906236320212) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[9],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[0],q[13];
rz(1.4265144426020613) q[0];
cx q[0],q[10];
rz(-1.4265144426020613) q[10];
cx q[0],q[10];
rz(1.4265144426020613) q[10];
cx q[10],q[5];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi) q[13];
rz(1.672535792453073) q[13];
cx q[13],q[0];
rz(-1.672535792453073) q[0];
cx q[13],q[0];
rz(1.672535792453073) q[0];
rz(1.2752262663017806) q[0];
sx q[0];
rz(4.570667653977831) q[0];
sx q[0];
rz(8.149551694467599) q[0];
cx q[0],q[18];
rz(0) q[18];
sx q[18];
rz(1.7114691694854536) q[18];
sx q[18];
rz(3*pi) q[18];
cx q[0],q[18];
rz(2.6443261234860995) q[0];
rz(pi/4) q[18];
rz(-1.3715383350258457) q[5];
cx q[10],q[5];
cx q[17],q[10];
cx q[10],q[17];
cx q[10],q[7];
id q[17];
rz(-0.4653228850187965) q[17];
sx q[17];
rz(6.069425312896955) q[17];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[5],q[4];
rz(0.7727836017639156) q[4];
cx q[5],q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[4],q[8];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[7],q[10];
cx q[10],q[7];
cx q[0],q[7];
rz(-2.6443261234860995) q[7];
cx q[0],q[7];
rz(-pi/4) q[0];
rz(pi/2) q[0];
rz(2.6443261234860995) q[7];
rz(pi) q[7];
rz(2.229695032646887) q[8];
cx q[4],q[8];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(-pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
id q[8];
cx q[8],q[12];
x q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(3.683058689712277) q[9];
rz(pi/2) q[9];
sx q[9];
rz(3.442524125680497) q[9];
sx q[9];
rz(5*pi/2) q[9];
cx q[9],q[1];
rz(2.34163788183991) q[1];
cx q[9],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(2.701053011831962) q[1];
sx q[1];
rz(5.7554277598595345) q[1];
sx q[1];
rz(14.905529225833849) q[1];
cx q[1],q[11];
rz(0.5172124623656192) q[11];
cx q[1],q[11];
rz(1.995432674855773) q[1];
rz(pi/2) q[1];
rz(-pi/2) q[11];
cx q[10],q[11];
rz(-pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[11];
rz(-pi/2) q[11];
cx q[4],q[1];
rz(0) q[1];
sx q[1];
rz(0.013215804439100864) q[1];
sx q[1];
rz(3*pi) q[1];
rz(0) q[4];
sx q[4];
rz(0.013215804439100864) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[4],q[1];
rz(-pi/2) q[1];
rz(-1.995432674855773) q[1];
rz(-1.5948941918848651) q[1];
rz(pi/2) q[1];
rz(-pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(2.091461551069401) q[4];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[13],q[9];
rz(pi/4) q[13];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[13],q[9];
rz(-pi/4) q[9];
cx q[13],q[9];
rz(pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[18],q[13];
rz(-pi/4) q[13];
cx q[18],q[13];
rz(pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/4) q[13];
cx q[17],q[13];
rz(-pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-pi/2) q[13];
rz(-pi/2) q[13];
cx q[17],q[7];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(2.923744334772953) q[18];
cx q[18],q[4];
rz(-2.923744334772953) q[4];
sx q[4];
rz(1.1878866730048663) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[18],q[4];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/4) q[18];
rz(0) q[4];
sx q[4];
rz(5.0952986341747195) q[4];
sx q[4];
rz(10.25706074447293) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(2.5650841820344596) q[4];
rz(-pi/2) q[7];
rz(pi/4) q[7];
rz(-pi/2) q[7];
rz(-pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
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
cx q[9],q[5];
rz(4.354647942986548) q[5];
cx q[9],q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
id q[5];
cx q[5],q[11];
rz(pi/2) q[11];
sx q[11];
cx q[15],q[11];
rz(pi/2) q[11];
x q[15];
rz(4.786107587347193) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[17],q[11];
cx q[11],q[17];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[3],q[15];
rz(-pi/4) q[15];
cx q[3],q[15];
rz(pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(-pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[16],q[9];
rz(2.0674236042796297) q[9];
cx q[16],q[9];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(0.16670959219355985) q[16];
rz(pi/2) q[16];
cx q[10],q[16];
rz(0) q[10];
sx q[10];
rz(1.3325810936251719) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[16];
sx q[16];
rz(1.3325810936251719) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[10],q[16];
rz(-pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
cx q[10],q[5];
rz(pi/4) q[10];
rz(-pi/2) q[16];
rz(-0.16670959219355985) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[0];
cx q[0],q[16];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-1.8142961151616137) q[0];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[18],q[16];
rz(-pi/4) q[16];
cx q[18],q[16];
rz(pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[17];
rz(4.375683695686064) q[17];
cx q[16],q[17];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(0.5708564540130807) q[16];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(-pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[10],q[5];
rz(-pi/4) q[5];
cx q[10],q[5];
rz(-pi/2) q[10];
rz(2.2315694426777095) q[10];
rz(pi/2) q[10];
rz(pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(3.8390339091306482) q[5];
cx q[5],q[0];
rz(-3.8390339091306482) q[0];
sx q[0];
rz(2.0754813907089966) q[0];
sx q[0];
rz(3*pi) q[0];
cx q[5],q[0];
rz(0) q[0];
sx q[0];
rz(4.20770391647059) q[0];
sx q[0];
rz(15.078107985061642) q[0];
rz(pi) q[0];
x q[0];
rz(3.8906844629209782) q[0];
rz(3.145098103809124) q[0];
rz(1.3609898399944353) q[5];
sx q[5];
rz(7.870314078713197) q[5];
sx q[5];
rz(15.16137899063739) q[5];
cx q[6],q[16];
rz(-0.5708564540130807) q[16];
cx q[6],q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(-pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[1];
rz(0) q[1];
sx q[1];
rz(2.25878333513409) q[1];
sx q[1];
rz(3*pi) q[1];
rz(0) q[9];
sx q[9];
rz(4.024401972045497) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[9],q[1];
rz(-pi/2) q[1];
rz(1.5948941918848651) q[1];
rz(2.142725772326862) q[1];
cx q[1],q[8];
rz(-2.142725772326862) q[8];
cx q[1],q[8];
rz(2.142725772326862) q[8];
rz(pi/2) q[8];
cx q[12],q[8];
cx q[8],q[12];
rz(-pi/2) q[12];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[10];
rz(0) q[10];
sx q[10];
rz(1.8612111100469544) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[8];
sx q[8];
rz(1.8612111100469544) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[8],q[10];
rz(-pi/2) q[10];
rz(-2.2315694426777095) q[10];
rz(pi/2) q[10];
rz(-pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(-1.1514228226737575) q[8];
cx q[0],q[8];
rz(-3.145098103809124) q[8];
sx q[8];
rz(1.3977541337413333) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[0],q[8];
rz(0) q[8];
sx q[8];
rz(4.885431173438253) q[8];
sx q[8];
rz(13.72129888725226) q[8];
rz(-pi/4) q[8];
rz(-pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
cx q[9],q[13];
rz(pi/2) q[13];
rz(-3.277442663310822) q[13];
rz(pi/2) q[13];
cx q[14],q[13];
rz(0) q[13];
sx q[13];
rz(0.8455510784920524) q[13];
sx q[13];
rz(3*pi) q[13];
rz(0) q[14];
sx q[14];
rz(5.437634228687534) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[14],q[13];
rz(-pi/2) q[13];
rz(3.277442663310822) q[13];
sx q[13];
cx q[11],q[13];
x q[11];
rz(1.1224054329012962) q[11];
cx q[11],q[2];
id q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(1.2440846948253452) q[13];
cx q[0],q[13];
rz(-1.2440846948253452) q[13];
cx q[0],q[13];
rz(pi/2) q[0];
sx q[0];
rz(5.586181721772052) q[0];
sx q[0];
rz(5*pi/2) q[0];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(2.9735889996617395) q[13];
rz(-pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(-2.0795487987144305) q[14];
rz(pi/2) q[14];
cx q[18],q[14];
rz(0) q[14];
sx q[14];
rz(1.8814704770849209) q[14];
sx q[14];
rz(3*pi) q[14];
rz(0) q[18];
sx q[18];
rz(4.401714830094665) q[18];
sx q[18];
rz(3*pi) q[18];
cx q[18],q[14];
rz(-pi/2) q[14];
rz(2.0795487987144305) q[14];
id q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(-pi/2) q[18];
rz(-1.1224054329012962) q[2];
sx q[2];
rz(1.4555580301171482) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[11],q[2];
cx q[11],q[6];
rz(0) q[2];
sx q[2];
rz(4.827627277062438) q[2];
sx q[2];
rz(11.53516685069522) q[2];
cx q[3],q[18];
cx q[10],q[3];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[3],q[10];
rz(0) q[10];
sx q[10];
rz(6.094438973308768) q[10];
sx q[10];
rz(3*pi) q[10];
rz(5.049559776598186) q[6];
cx q[11],q[6];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
cx q[8],q[6];
cx q[6],q[8];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/4) q[9];
cx q[1],q[9];
cx q[1],q[12];
rz(5.024336756713969) q[1];
rz(0.5653210360998863) q[1];
cx q[1],q[2];
rz(pi/2) q[12];
cx q[12],q[15];
cx q[15],q[12];
rz(0.10123438520945187) q[12];
sx q[12];
rz(7.309017660300134) q[12];
sx q[12];
rz(14.123506611370457) q[12];
id q[12];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[18],q[15];
rz(4.059340665841494) q[15];
cx q[18],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(-pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-0.5653210360998863) q[2];
cx q[1],q[2];
rz(0.5653210360998863) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/4) q[2];
cx q[1],q[2];
rz(-pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-pi/2) q[2];
rz(-pi/4) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(-pi/2) q[9];
cx q[9],q[4];
rz(-2.5650841820344596) q[4];
cx q[9],q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-pi/2) q[4];
cx q[5],q[4];
rz(pi/2) q[4];
rz(3.4053482804475066) q[4];
rz(pi/2) q[4];
cx q[5],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/4) q[5];
cx q[5],q[14];
rz(-pi/4) q[14];
cx q[5],q[14];
rz(pi/4) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[13];
rz(-2.9735889996617395) q[13];
cx q[14],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[15],q[5];
rz(4.598040368908521) q[5];
cx q[15],q[5];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
cx q[7],q[4];
rz(0) q[4];
sx q[4];
rz(1.2290947832816874) q[4];
sx q[4];
rz(3*pi) q[4];
rz(0) q[7];
sx q[7];
rz(1.2290947832816874) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[7],q[4];
rz(-pi/2) q[4];
rz(-3.4053482804475066) q[4];
rz(0) q[4];
sx q[4];
rz(4.45709785146407) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[3],q[4];
rz(0) q[4];
sx q[4];
rz(1.8260874557155164) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[3],q[4];
rz(-pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
cx q[16],q[7];
rz(5.465423428391425) q[7];
cx q[16],q[7];
rz(4.230259956959507) q[9];
rz(pi/2) q[9];
cx q[17],q[9];
rz(0) q[17];
sx q[17];
rz(2.0685077243362535) q[17];
sx q[17];
rz(3*pi) q[17];
rz(0) q[9];
sx q[9];
rz(2.0685077243362535) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[17],q[9];
rz(-pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(-pi/4) q[17];
rz(-pi/2) q[9];
rz(-4.230259956959507) q[9];
rz(4.273396415293824) q[9];
sx q[9];
rz(8.395810294523422) q[9];
sx q[9];
rz(15.06261357743171) q[9];
rz(pi/4) q[9];
cx q[9],q[18];
rz(-pi/4) q[18];
cx q[9],q[18];
rz(pi/4) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
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
