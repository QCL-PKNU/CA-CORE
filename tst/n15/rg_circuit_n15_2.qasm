OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
creg c[15];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[0],q[1];
rz(pi/4) q[0];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[0],q[1];
rz(-pi/4) q[1];
cx q[0],q[1];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
sx q[2];
rz(-pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(2.0494473625999214) q[5];
rz(pi/2) q[5];
sx q[5];
rz(7.777728840389867) q[5];
sx q[5];
rz(5*pi/2) q[5];
rz(1.3732379526270482) q[5];
x q[6];
rz(0) q[6];
sx q[6];
rz(3.441888450986765) q[6];
sx q[6];
rz(3*pi) q[6];
rz(0) q[6];
sx q[6];
rz(7.984633941065047) q[6];
sx q[6];
rz(3*pi) q[6];
rz(-pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(0.9762382710067925) q[6];
rz(0.025942689162868238) q[7];
rz(0.44053943882393887) q[8];
rz(pi/2) q[8];
cx q[3],q[8];
rz(0) q[3];
sx q[3];
rz(1.242699826401468) q[3];
sx q[3];
rz(3*pi) q[3];
rz(0) q[8];
sx q[8];
rz(1.242699826401468) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[3],q[8];
rz(-pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(-pi/2) q[8];
rz(-0.44053943882393887) q[8];
rz(2.033752330621379) q[9];
cx q[9],q[7];
rz(-2.033752330621379) q[7];
sx q[7];
rz(2.80796478581283) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[9],q[7];
rz(0) q[7];
sx q[7];
rz(3.475220521366756) q[7];
sx q[7];
rz(11.43258760222789) q[7];
rz(-pi/2) q[7];
cx q[8],q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(0) q[9];
sx q[9];
rz(4.532545946301446) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[0],q[9];
rz(0) q[9];
sx q[9];
rz(1.7506393608781403) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[0],q[9];
rz(-3.3253269657589968) q[0];
sx q[0];
rz(5.95089029411156) q[0];
sx q[0];
rz(12.750104926528376) q[0];
sx q[9];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(1.3095151302219985) q[10];
sx q[10];
rz(4.123844992581185) q[10];
sx q[10];
rz(12.706441514241682) q[10];
rz(pi/2) q[10];
cx q[10],q[9];
x q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[0],q[10];
rz(pi/4) q[0];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[0],q[10];
rz(-pi/4) q[10];
cx q[0],q[10];
rz(4.482338031140244) q[0];
rz(2.2003316214547493) q[0];
rz(pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(2.058256053418207) q[9];
cx q[0],q[9];
rz(-2.2003316214547493) q[9];
sx q[9];
rz(3.110118871328506) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[0],q[9];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(0) q[9];
sx q[9];
rz(3.17306643585108) q[9];
sx q[9];
rz(9.566853528805922) q[9];
rz(pi/2) q[11];
cx q[11],q[2];
cx q[1],q[2];
x q[11];
rz(3.156029412773336) q[11];
rz(3.850997316479709) q[2];
cx q[1],q[2];
rz(2.8108442610057986) q[1];
cx q[1],q[5];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-2.8108442610057986) q[5];
sx q[5];
rz(2.399521564287979) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[1],q[5];
rz(3.207120308498801) q[1];
rz(0) q[5];
sx q[5];
rz(3.883663742891607) q[5];
sx q[5];
rz(10.86238426914813) q[5];
rz(pi/4) q[5];
rz(-pi/4) q[5];
rz(2.3227932552148935) q[5];
cx q[5],q[9];
rz(-2.3227932552148935) q[9];
cx q[5],q[9];
sx q[5];
rz(2.3227932552148935) q[9];
rz(1.6307254439642493) q[9];
sx q[12];
rz(4.550017182867426) q[12];
rz(3.9203332834038562) q[12];
rz(pi/2) q[12];
rz(-pi/2) q[13];
cx q[4],q[13];
rz(pi/2) q[13];
rz(5.9497585756463005) q[13];
rz(0.6541326065123458) q[13];
rz(0.20885456269826763) q[4];
cx q[11],q[4];
rz(-3.156029412773336) q[4];
sx q[4];
rz(1.664755900436376) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[11],q[4];
id q[11];
rz(3.8093663886236278) q[11];
sx q[11];
rz(4.165615589999868) q[11];
sx q[11];
rz(10.53001007166291) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[11],q[10];
cx q[10],q[11];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(0) q[4];
sx q[4];
rz(4.618429406743211) q[4];
sx q[4];
rz(12.371952810844448) q[4];
rz(pi/4) q[4];
cx q[4],q[7];
rz(-pi/4) q[7];
cx q[4],q[7];
rz(0.032648263645233176) q[4];
rz(pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi) q[7];
x q[7];
cx q[7],q[6];
rz(-0.9762382710067925) q[6];
cx q[7],q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(2.2960340236565826) q[7];
sx q[7];
rz(4.1216480915531815) q[7];
sx q[7];
rz(13.177818774114812) q[7];
rz(pi/2) q[7];
cx q[8],q[13];
rz(-0.6541326065123458) q[13];
cx q[8],q[13];
rz(2.199369954387788) q[13];
cx q[1],q[13];
rz(-3.207120308498801) q[13];
sx q[13];
rz(1.1386478272104346) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[1],q[13];
rz(pi/2) q[1];
sx q[1];
rz(8.696665457418357) q[1];
sx q[1];
rz(5*pi/2) q[1];
rz(0) q[13];
sx q[13];
rz(5.144537479969152) q[13];
sx q[13];
rz(10.432528314880392) q[13];
rz(0) q[13];
sx q[13];
rz(5.8295374678551) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[4],q[13];
rz(0) q[13];
sx q[13];
rz(0.4536478393244865) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[4],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[0];
rz(1.2427474152670503) q[0];
cx q[13],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(1.7857905540684929) q[0];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(0.14302223671624914) q[4];
rz(3.9718746065664563) q[8];
sx q[8];
rz(8.74174282501956) q[8];
sx q[8];
rz(15.509507068055097) q[8];
x q[14];
cx q[14],q[3];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[2];
rz(1.7398211445716112) q[2];
cx q[14],q[2];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(1.992249957590332) q[14];
sx q[14];
rz(5.51387338011887) q[14];
sx q[14];
rz(12.131037974646816) q[14];
rz(0) q[14];
sx q[14];
rz(6.279134786140864) q[14];
sx q[14];
rz(3*pi) q[14];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[12];
rz(0) q[12];
sx q[12];
rz(0.15122338733596052) q[12];
sx q[12];
rz(3*pi) q[12];
rz(0) q[3];
sx q[3];
rz(0.15122338733596052) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[3],q[12];
rz(-pi/2) q[12];
rz(-3.9203332834038562) q[12];
cx q[12],q[2];
rz(4.7509772403990045) q[2];
cx q[12],q[2];
rz(-pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
cx q[2],q[11];
cx q[11],q[2];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[11],q[7];
rz(pi/2) q[2];
sx q[2];
rz(6.244586320323932) q[2];
sx q[2];
rz(5*pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(3.1038584687994284) q[3];
sx q[3];
rz(8.510045195089347) q[3];
sx q[3];
rz(12.066309616844489) q[3];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[12];
rz(6.069989467254638) q[12];
cx q[3],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
cx q[12],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/4) q[12];
cx q[12],q[10];
rz(-pi/4) q[10];
cx q[12],q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
x q[10];
rz(pi/4) q[12];
cx q[12],q[13];
rz(-pi/4) q[13];
cx q[12],q[13];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(5.077490859019068) q[13];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
cx q[3],q[1];
rz(3.316336466867903) q[1];
cx q[3],q[1];
sx q[3];
rz(0.7511371762551657) q[3];
cx q[6],q[10];
cx q[10],q[6];
cx q[7],q[11];
rz(pi) q[11];
x q[11];
rz(0) q[11];
sx q[11];
rz(3.6383861130440645) q[11];
sx q[11];
rz(3*pi) q[11];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[3],q[7];
rz(-0.7511371762551657) q[7];
cx q[3],q[7];
rz(-1.7137502501200232) q[3];
rz(0.7511371762551657) q[7];
rz(pi/4) q[7];
rz(6.09050655753291) q[7];
cx q[8],q[14];
rz(0) q[14];
sx q[14];
rz(0.004050521038721833) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[8],q[14];
rz(-pi/2) q[14];
rz(-pi/2) q[14];
cx q[1],q[14];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(1.6353863767522776) q[1];
sx q[1];
rz(6.527088340870209) q[1];
sx q[1];
rz(13.324094129676405) q[1];
sx q[1];
rz(-pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[14];
rz(-pi/2) q[14];
cx q[13],q[14];
cx q[14],q[13];
rz(0) q[13];
sx q[13];
rz(6.79869121239275) q[13];
sx q[13];
rz(3*pi) q[13];
rz(-pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(1.279818665083016) q[8];
cx q[8],q[4];
rz(-1.279818665083016) q[4];
sx q[4];
rz(2.954650464658558) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[8],q[4];
rz(0) q[4];
sx q[4];
rz(3.3285348425210284) q[4];
sx q[4];
rz(10.561574389136146) q[4];
cx q[8],q[0];
rz(-1.7857905540684929) q[0];
cx q[8],q[0];
cx q[0],q[12];
rz(pi/4) q[0];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[0],q[12];
rz(-pi/4) q[12];
cx q[0],q[12];
rz(3.494905301739077) q[0];
sx q[0];
rz(5.620014385474223) q[0];
sx q[0];
rz(15.277884024450707) q[0];
rz(pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[11];
rz(0) q[11];
sx q[11];
rz(2.644799194135522) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[12],q[11];
rz(0.0716850524115058) q[11];
sx q[11];
rz(2.6353795202326378) q[11];
rz(pi/2) q[12];
cx q[8],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[6],q[2];
cx q[2],q[6];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(0) q[6];
sx q[6];
rz(9.029687772103195) q[6];
sx q[6];
rz(3*pi) q[6];
rz(pi) q[8];
rz(2.2178200752917463) q[8];
cx q[8],q[0];
rz(-2.2178200752917463) q[0];
cx q[8],q[0];
rz(2.2178200752917463) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/4) q[0];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/4) q[8];
cx q[6],q[8];
rz(-1.3439506480667895) q[6];
sx q[6];
rz(2.1727093668092383) q[6];
rz(-pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(2.9689712256350345) q[8];
rz(pi/2) q[8];
cx q[9],q[4];
rz(-1.6307254439642493) q[4];
cx q[9],q[4];
rz(1.6307254439642493) q[4];
cx q[4],q[5];
cx q[5],q[4];
rz(1.0822193726542177) q[4];
rz(4.346308971273245) q[4];
cx q[4],q[3];
rz(-4.346308971273245) q[3];
sx q[3];
rz(2.8599853900932626) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[4],q[3];
rz(0) q[3];
sx q[3];
rz(3.4231999170863237) q[3];
sx q[3];
rz(15.484837182162646) q[3];
sx q[3];
cx q[12],q[3];
x q[12];
rz(-pi/4) q[12];
rz(-pi/2) q[12];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[13],q[3];
rz(5.249063245108625) q[3];
cx q[13],q[3];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(-pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(0.6995423212665821) q[4];
sx q[4];
rz(4.154027281358644) q[4];
sx q[4];
rz(8.725235639502797) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(2.4054781130296337) q[4];
rz(3.1506194999679353) q[5];
sx q[5];
rz(4.176387503982492) q[5];
sx q[5];
rz(11.009838598897758) q[5];
cx q[5],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-2.3643101497879035) q[5];
rz(pi/2) q[5];
cx q[1],q[5];
rz(0) q[1];
sx q[1];
rz(5.007025825909395) q[1];
sx q[1];
rz(3*pi) q[1];
rz(0) q[5];
sx q[5];
rz(1.2761594812701917) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[1],q[5];
rz(-pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/4) q[1];
rz(-pi/2) q[5];
rz(2.3643101497879035) q[5];
cx q[5],q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[5];
cx q[7],q[4];
rz(-2.4054781130296337) q[4];
cx q[7],q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[1],q[4];
rz(-pi/4) q[4];
cx q[1],q[4];
rz(pi/2) q[1];
sx q[1];
rz(3.997381502605176) q[1];
sx q[1];
rz(5*pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(2.848539246787982) q[4];
rz(0) q[7];
sx q[7];
rz(6.259559278886734) q[7];
sx q[7];
rz(3*pi) q[7];
sx q[9];
cx q[9],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[2];
rz(1.0931940099629696) q[2];
cx q[10],q[2];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[0];
rz(-pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/2) q[0];
cx q[0],q[7];
rz(-3.131739349258752) q[10];
rz(pi/2) q[10];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[11],q[2];
rz(pi/4) q[11];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[11],q[2];
rz(-pi/4) q[2];
cx q[11],q[2];
rz(pi/4) q[11];
rz(-pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[11],q[8];
rz(0) q[11];
sx q[11];
rz(0.5011398925718082) q[11];
sx q[11];
rz(3*pi) q[11];
rz(pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(0.7190212590350484) q[2];
sx q[2];
rz(3.6607630198826135) q[2];
sx q[2];
rz(8.70575670173433) q[2];
rz(pi/4) q[2];
cx q[2],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(2.8099266900317628) q[1];
rz(4.36354979976735) q[2];
cx q[3],q[10];
rz(0) q[10];
sx q[10];
rz(0.5324671622198554) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[3];
sx q[3];
rz(5.75071814495973) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[3],q[10];
rz(-pi/2) q[10];
rz(3.131739349258752) q[10];
rz(-pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
cx q[3],q[6];
rz(3.417914274028177) q[6];
cx q[3],q[6];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(0) q[7];
sx q[7];
rz(0.023626028292852208) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[0],q[7];
rz(-2.1598037755935313) q[0];
cx q[4],q[0];
rz(-2.848539246787982) q[0];
sx q[0];
rz(2.2674960279185488) q[0];
sx q[0];
rz(3*pi) q[0];
cx q[4],q[0];
rz(0) q[0];
sx q[0];
rz(4.0156892792610375) q[0];
sx q[0];
rz(14.433120983150893) q[0];
rz(0.4392319985380364) q[0];
sx q[0];
rz(5.298804027626467) q[0];
sx q[0];
rz(8.985545962231344) q[0];
sx q[4];
sx q[7];
cx q[5],q[7];
x q[5];
rz(4.330464654124419) q[5];
rz(pi/2) q[5];
rz(-pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[5];
rz(0) q[5];
sx q[5];
rz(0.23028578026099833) q[5];
sx q[5];
rz(3*pi) q[5];
rz(0) q[7];
sx q[7];
rz(0.23028578026099833) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[7],q[5];
rz(-pi/2) q[5];
rz(-4.330464654124419) q[5];
cx q[5],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/4) q[5];
cx q[5],q[3];
rz(-pi/4) q[3];
cx q[5],q[3];
rz(pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(0) q[5];
sx q[5];
rz(3.6182347224014677) q[5];
sx q[5];
rz(3*pi) q[5];
x q[5];
rz(-pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(0) q[8];
sx q[8];
rz(0.5011398925718082) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[11],q[8];
rz(-pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/2) q[8];
rz(-2.9689712256350345) q[8];
rz(pi/2) q[8];
cx q[8],q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[7],q[4];
rz(1.6177010423361813) q[4];
cx q[7],q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(1.4516624480039937) q[7];
cx q[7],q[4];
rz(-1.4516624480039937) q[4];
cx q[7],q[4];
rz(1.4516624480039937) q[4];
rz(1.3487808847990643) q[4];
rz(0) q[4];
sx q[4];
rz(3.8185915211273453) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[5],q[4];
rz(0) q[4];
sx q[4];
rz(2.464593786052241) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[5],q[4];
rz(pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[5];
rz(-pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
x q[8];
rz(pi/4) q[8];
rz(2.7942362771431304) q[8];
rz(0.23148733415416445) q[9];
cx q[9],q[14];
rz(2.995138365141518) q[14];
cx q[9],q[14];
rz(1.8026475805356108) q[14];
cx q[13],q[14];
rz(-1.8026475805356108) q[14];
cx q[13],q[14];
rz(-pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[10];
rz(3.115248281155615) q[10];
cx q[13],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/4) q[10];
cx q[10],q[12];
rz(-pi/4) q[12];
cx q[10],q[12];
rz(5.407980869319536) q[10];
rz(3.7116550788879192) q[10];
sx q[10];
rz(8.732794607159445) q[10];
sx q[10];
rz(12.636235396631273) q[10];
rz(2.8881769406021864) q[10];
rz(pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(0) q[13];
sx q[13];
rz(8.419359680496658) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[13],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/2) q[11];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(0) q[9];
sx q[9];
rz(7.321266439881144) q[9];
sx q[9];
rz(3*pi) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[14],q[9];
rz(4.011511800982847) q[9];
cx q[14],q[9];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(0.8434121633442591) q[14];
cx q[14],q[6];
rz(-0.8434121633442591) q[6];
cx q[14],q[6];
rz(-0.6108631851974433) q[14];
cx q[2],q[14];
rz(-4.36354979976735) q[14];
sx q[14];
rz(1.5877440229848023) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[2],q[14];
rz(0) q[14];
sx q[14];
rz(4.695441284194784) q[14];
sx q[14];
rz(14.399190945734173) q[14];
sx q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[2],q[11];
cx q[10],q[2];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[11];
rz(-2.8881769406021864) q[2];
cx q[10],q[2];
rz(pi/2) q[10];
sx q[10];
rz(5.912211542254063) q[10];
sx q[10];
rz(5*pi/2) q[10];
sx q[10];
rz(2.8881769406021864) q[2];
cx q[5],q[10];
x q[5];
rz(0.8434121633442591) q[6];
cx q[1],q[6];
rz(-2.8099266900317628) q[6];
cx q[1],q[6];
cx q[1],q[13];
rz(5.921252059841157) q[13];
cx q[1],q[13];
rz(-pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
cx q[13],q[2];
cx q[2],q[13];
cx q[13],q[2];
rz(-1.267055793939831) q[2];
rz(2.8099266900317628) q[6];
cx q[6],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/4) q[6];
cx q[6],q[12];
rz(-pi/4) q[12];
cx q[6],q[12];
rz(pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi) q[12];
cx q[6],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[7],q[1];
rz(2.02889097587427) q[1];
cx q[7],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(2.6379253787188666) q[1];
cx q[1],q[14];
rz(-2.6379253787188666) q[14];
cx q[1],q[14];
rz(2.6379253787188666) q[14];
rz(3.6971847891322267) q[14];
rz(1.4770511548163212) q[14];
cx q[14],q[2];
rz(-1.4770511548163212) q[2];
sx q[2];
rz(0.43535354199610055) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[14],q[2];
rz(pi/2) q[14];
rz(0) q[2];
sx q[2];
rz(5.847831765183486) q[2];
sx q[2];
rz(12.16888490952553) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(1.6986308425105556) q[2];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[6],q[7];
rz(pi/4) q[6];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[6],q[7];
rz(-pi/4) q[7];
cx q[6],q[7];
rz(pi) q[6];
rz(pi) q[6];
x q[6];
rz(pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
x q[7];
rz(-pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
cx q[9],q[0];
cx q[0],q[9];
cx q[8],q[0];
rz(-2.7942362771431304) q[0];
cx q[8],q[0];
rz(2.7942362771431304) q[0];
id q[0];
cx q[8],q[11];
rz(-pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/4) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[11],q[14];
cx q[14],q[11];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(4.904972205028129) q[8];
sx q[8];
rz(5*pi/2) q[8];
rz(0) q[8];
sx q[8];
rz(6.5015031508504055) q[8];
sx q[8];
rz(3*pi) q[8];
rz(1.9838121469798704) q[9];
cx q[9],q[3];
rz(-1.9838121469798704) q[3];
cx q[9],q[3];
rz(1.9838121469798704) q[3];
rz(pi/2) q[3];
sx q[9];
cx q[3],q[9];
x q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/4) q[3];
cx q[12],q[3];
sx q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[13],q[3];
rz(5.628960366248494) q[3];
cx q[13],q[3];
cx q[10],q[13];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(0.6748653078867332) q[3];
cx q[5],q[3];
rz(-0.6748653078867332) q[3];
cx q[5],q[3];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(1.18533597045615) q[9];
cx q[0],q[9];
rz(-1.18533597045615) q[9];
cx q[0],q[9];
cx q[1],q[0];
cx q[0],q[1];
cx q[1],q[0];
cx q[0],q[2];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[12],q[1];
rz(2.4048774353314757) q[1];
cx q[12],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-1.6986308425105556) q[2];
cx q[0],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/4) q[9];
rz(-5.701092211332348) q[9];
rz(pi/2) q[9];
cx q[7],q[9];
rz(0) q[7];
sx q[7];
rz(3.768359373012237) q[7];
sx q[7];
rz(3*pi) q[7];
rz(0) q[9];
sx q[9];
rz(2.5148259341673493) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[7],q[9];
rz(-pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(-pi/2) q[9];
rz(5.701092211332348) q[9];
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