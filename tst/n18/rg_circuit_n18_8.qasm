OPENQASM 2.0;
include "qelib1.inc";
qreg q[18];
creg c[18];
rz(-pi/2) q[0];
rz(0.806588584001059) q[1];
rz(pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(5.302899899373819) q[3];
sx q[3];
rz(5*pi/2) q[3];
rz(pi) q[3];
x q[3];
rz(-1.5095631321886718) q[3];
rz(-pi/4) q[4];
rz(3.106730975583762) q[5];
rz(pi/2) q[5];
x q[6];
rz(-4.081104225408994) q[6];
rz(pi/2) q[6];
cx q[7],q[2];
cx q[2],q[7];
cx q[7],q[2];
cx q[2],q[4];
cx q[4],q[2];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[5];
rz(0) q[5];
sx q[5];
rz(0.23654296369543548) q[5];
sx q[5];
rz(3*pi) q[5];
rz(0) q[8];
sx q[8];
rz(0.23654296369543548) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[8],q[5];
rz(-pi/2) q[5];
rz(-3.106730975583762) q[5];
rz(1.0811302853940257) q[5];
sx q[5];
rz(4.801840920900376) q[5];
sx q[5];
rz(14.434255020509802) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[2],q[5];
rz(-pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/4) q[5];
rz(-pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(-pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(0) q[9];
sx q[9];
rz(3.3815466826802014) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[1],q[10];
rz(-0.806588584001059) q[10];
cx q[1],q[10];
cx q[1],q[9];
rz(0.806588584001059) q[10];
rz(-pi/2) q[10];
rz(2.8048689638664364) q[9];
cx q[1],q[9];
rz(4.3771844039608645) q[1];
sx q[1];
rz(7.813116478714841) q[1];
sx q[1];
rz(11.294210336412183) q[1];
rz(pi) q[9];
x q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(-1.80406361311787) q[11];
sx q[11];
rz(3.6750215019437835) q[11];
sx q[11];
rz(11.22884157388725) q[11];
rz(-pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[11],q[6];
rz(0) q[11];
sx q[11];
rz(4.937693964956401) q[11];
sx q[11];
rz(3*pi) q[11];
rz(0) q[6];
sx q[6];
rz(1.345491342223185) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[11],q[6];
rz(-pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(-pi/2) q[6];
rz(4.081104225408994) q[6];
rz(pi/4) q[6];
rz(-4.612175618570632) q[12];
sx q[12];
rz(3.5736981041906617) q[12];
sx q[12];
rz(14.036953579340011) q[12];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[14],q[0];
rz(pi/2) q[0];
rz(-2.7498841350890433) q[0];
rz(pi/2) q[0];
cx q[12],q[0];
rz(0) q[0];
sx q[0];
rz(2.808518319228875) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[12];
sx q[12];
rz(3.474666987950711) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[12],q[0];
rz(-pi/2) q[0];
rz(2.7498841350890433) q[0];
rz(4.065880265731183) q[0];
rz(4.937374602352524) q[0];
rz(-pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(-pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[4],q[12];
cx q[12],q[4];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[4],q[12];
rz(0.8363916306117595) q[12];
cx q[4],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(pi) q[4];
id q[4];
cx q[8],q[14];
rz(3.9368597367900677) q[14];
cx q[8],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/4) q[14];
rz(-1.6263958248365151) q[14];
sx q[14];
rz(9.237204351712982) q[14];
sx q[14];
rz(11.051173785605894) q[14];
rz(pi) q[14];
rz(4.429720573753637) q[14];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(0.21428509118590425) q[8];
cx q[8],q[11];
rz(-0.21428509118590425) q[11];
cx q[8],q[11];
rz(0.21428509118590425) q[11];
rz(4.339194075873286) q[11];
cx q[11],q[3];
rz(-4.339194075873286) q[3];
sx q[3];
rz(1.4364888393864286) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[11],q[3];
rz(0) q[3];
sx q[3];
rz(4.846696467793158) q[3];
sx q[3];
rz(15.273535168831337) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[2];
rz(4.715431710721264) q[2];
cx q[8],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(0.3389736102236419) q[2];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[15],q[16];
rz(pi/4) q[15];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[15],q[16];
rz(-pi/4) q[16];
cx q[15],q[16];
rz(pi) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[16];
cx q[6],q[10];
rz(-pi/4) q[10];
cx q[6],q[10];
cx q[1],q[6];
rz(pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(0.39682547127748713) q[10];
sx q[10];
rz(4.208617202204921) q[10];
sx q[10];
rz(9.562978380761432) q[10];
rz(0.34033825974374743) q[6];
cx q[1],q[6];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[6];
rz(-0.6262235006750126) q[6];
sx q[6];
rz(5.901371059123782) q[6];
rz(pi/2) q[6];
rz(1.491295550488914) q[17];
cx q[17],q[13];
rz(-1.491295550488914) q[13];
cx q[17],q[13];
rz(1.491295550488914) q[13];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(-pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[7],q[13];
rz(1.8181491639805696) q[13];
cx q[7],q[13];
rz(pi/2) q[13];
cx q[15],q[13];
cx q[13],q[15];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(1.4985099855239012) q[13];
cx q[15],q[9];
cx q[12],q[15];
cx q[15],q[12];
rz(2.4125225821324197) q[12];
cx q[12],q[15];
rz(-2.4125225821324197) q[15];
cx q[12],q[15];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(2.4125225821324197) q[15];
rz(pi/4) q[15];
cx q[16],q[13];
rz(-1.4985099855239012) q[13];
cx q[16],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-pi/2) q[13];
rz(0) q[16];
sx q[16];
rz(3.2061515223185504) q[16];
sx q[16];
rz(3*pi) q[16];
rz(3.3216215156733013) q[7];
rz(pi/2) q[7];
cx q[17],q[7];
rz(0) q[17];
sx q[17];
rz(3.0267752118016626) q[17];
sx q[17];
rz(3*pi) q[17];
rz(0) q[7];
sx q[7];
rz(3.0267752118016626) q[7];
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
rz(-0.15558821755518326) q[17];
sx q[17];
rz(8.743360325729537) q[17];
sx q[17];
rz(9.580366178324562) q[17];
cx q[2],q[17];
rz(-0.3389736102236419) q[17];
cx q[2],q[17];
rz(0.3389736102236419) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/4) q[17];
id q[2];
rz(-pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[5],q[17];
rz(-pi/4) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(-pi/2) q[17];
rz(4.991132447803797) q[5];
sx q[5];
rz(4.994736592601339) q[5];
sx q[5];
rz(15.06145411618318) q[5];
rz(pi/2) q[5];
rz(-pi/2) q[7];
rz(-3.3216215156733013) q[7];
rz(-0.4152937486740451) q[7];
cx q[0],q[7];
rz(-4.937374602352524) q[7];
sx q[7];
rz(1.1709630364379515) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[0],q[7];
rz(pi/4) q[0];
cx q[0],q[3];
rz(-pi/4) q[3];
cx q[0],q[3];
cx q[0],q[13];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(0.9379765555261975) q[0];
rz(pi/2) q[13];
rz(1.9410639064106585) q[13];
cx q[17],q[13];
rz(-1.9410639064106585) q[13];
cx q[17],q[13];
rz(-0.7692586541844242) q[13];
rz(-pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(3.7020529626161203) q[3];
sx q[3];
rz(5*pi/2) q[3];
rz(pi/2) q[3];
rz(0) q[7];
sx q[7];
rz(5.1122222707416345) q[7];
sx q[7];
rz(14.777446311795948) q[7];
cx q[11],q[7];
cx q[7],q[11];
cx q[10],q[11];
cx q[11],q[10];
cx q[10],q[11];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[6];
cx q[11],q[4];
rz(0) q[11];
sx q[11];
rz(6.1628936501807505) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[4],q[11];
rz(0) q[11];
sx q[11];
rz(0.1202916569988357) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[4],q[11];
rz(-pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(1.9600043396078097) q[4];
cx q[6],q[10];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(0) q[6];
sx q[6];
rz(5.309991262754336) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[15],q[6];
rz(0) q[6];
sx q[6];
rz(0.9731940444252505) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[15],q[6];
rz(0) q[6];
sx q[6];
rz(8.624751677768177) q[6];
sx q[6];
rz(3*pi) q[6];
rz(1.7938499228437919) q[7];
sx q[7];
rz(5.288700688207068) q[7];
sx q[7];
rz(10.83121193364534) q[7];
cx q[14],q[7];
cx q[7],q[14];
cx q[14],q[10];
rz(-pi/2) q[10];
rz(-pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(0.17335147056888903) q[7];
rz(4.875466805309266) q[7];
cx q[7],q[13];
rz(-4.875466805309266) q[13];
sx q[13];
rz(1.2481048818586502) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[7],q[13];
rz(0) q[13];
sx q[13];
rz(5.0350804253209365) q[13];
sx q[13];
rz(15.06950342026307) q[13];
rz(4.86946976347277) q[13];
cx q[6],q[13];
rz(1.2748675449799534) q[13];
sx q[13];
rz(6.988760918619969) q[13];
sx q[13];
rz(13.178440355620559) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(3.0425206120493886) q[7];
cx q[8],q[16];
rz(0) q[16];
sx q[16];
rz(3.077033784861036) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[8],q[16];
rz(1.1970819952222271) q[16];
rz(0) q[16];
sx q[16];
rz(8.68623644804513) q[16];
sx q[16];
rz(3*pi) q[16];
rz(-pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[17],q[16];
rz(5.939076982543002) q[16];
cx q[17],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(5.276271629534404) q[16];
rz(pi/2) q[16];
cx q[11],q[16];
rz(0) q[11];
sx q[11];
rz(0.8626300169151251) q[11];
sx q[11];
rz(3*pi) q[11];
rz(0) q[16];
sx q[16];
rz(0.8626300169151251) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[11],q[16];
rz(-pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(2.7938345543561915) q[11];
sx q[11];
rz(-pi/2) q[16];
rz(-5.276271629534404) q[16];
rz(3.6277485764233814) q[16];
sx q[16];
rz(5.3296558870722555) q[16];
sx q[16];
rz(11.25950441655942) q[16];
rz(-2.3039530269887236) q[16];
sx q[16];
rz(9.197157415294111) q[16];
sx q[16];
rz(11.728730987758103) q[16];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
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
sx q[1];
cx q[3],q[1];
rz(1.7659430726969756) q[1];
rz(2.6708207332664395) q[1];
x q[3];
cx q[3],q[0];
rz(-0.9379765555261975) q[0];
cx q[3],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/2) q[0];
cx q[15],q[0];
rz(pi/2) q[0];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[3],q[4];
rz(-1.9600043396078097) q[4];
cx q[3],q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-2.5209060572902615) q[4];
sx q[4];
rz(8.110058762228423) q[4];
sx q[4];
rz(11.94568401805964) q[4];
rz(3.994546146667044) q[4];
sx q[4];
rz(7.075537034443538) q[4];
sx q[4];
rz(12.269548097060769) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[12];
rz(5.657335645437154) q[12];
cx q[8],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(1.7517699310330082) q[12];
rz(pi/2) q[12];
sx q[12];
rz(9.408352120807947) q[12];
sx q[12];
rz(5*pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(5.994963635463543) q[8];
rz(-pi/2) q[8];
cx q[5],q[8];
cx q[5],q[3];
rz(4.292280182505546) q[3];
cx q[5],q[3];
rz(pi/2) q[3];
rz(0) q[3];
sx q[3];
rz(4.1315254859987665) q[3];
sx q[3];
rz(3*pi) q[3];
rz(-1.333702453821373) q[5];
sx q[5];
rz(5.505245654324176) q[5];
rz(pi/2) q[8];
rz(pi/4) q[8];
cx q[8],q[12];
rz(-pi/4) q[12];
cx q[8],q[12];
rz(pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi/2) q[12];
cx q[12],q[6];
rz(pi/4) q[12];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[12],q[6];
rz(-pi/4) q[6];
cx q[12],q[6];
rz(pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[12],q[6];
rz(pi/2) q[12];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(-pi/2) q[6];
rz(-pi/4) q[6];
sx q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(1.2316780163744616) q[9];
sx q[9];
rz(3.9219004535265722) q[9];
sx q[9];
rz(11.80760770372868) q[9];
rz(-pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[2],q[9];
rz(3.0726558454533963) q[9];
cx q[2],q[9];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(1.6342773255091894) q[2];
cx q[1],q[2];
rz(-2.6708207332664395) q[2];
sx q[2];
rz(0.09060846947542389) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[1],q[2];
rz(0.45652643786614144) q[1];
rz(pi/2) q[1];
cx q[14],q[1];
rz(0) q[1];
sx q[1];
rz(0.9683360046476914) q[1];
sx q[1];
rz(3*pi) q[1];
rz(0) q[14];
sx q[14];
rz(0.9683360046476914) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[14],q[1];
rz(-pi/2) q[1];
rz(-0.45652643786614144) q[1];
rz(0.8703695815578814) q[1];
rz(-pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
cx q[0],q[14];
rz(4.543627372165834) q[14];
cx q[0],q[14];
rz(1.9188007078588614) q[0];
sx q[0];
rz(6.510954125945107) q[0];
rz(-2.193090212466163) q[0];
sx q[0];
rz(8.980996499656367) q[0];
sx q[0];
rz(11.617868173235543) q[0];
rz(pi/2) q[0];
rz(pi/2) q[14];
cx q[14],q[8];
x q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(0) q[2];
sx q[2];
rz(6.192576837704163) q[2];
sx q[2];
rz(10.46132136852663) q[2];
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
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[7],q[2];
rz(-3.0425206120493886) q[2];
cx q[7],q[2];
rz(3.0425206120493886) q[2];
cx q[2],q[1];
rz(-0.8703695815578814) q[1];
cx q[2],q[1];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[10],q[7];
rz(0.35094641401973453) q[7];
cx q[10],q[7];
rz(pi/2) q[10];
cx q[10],q[3];
rz(0) q[3];
sx q[3];
rz(2.1516598211808193) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[10],q[3];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi/2) q[7];
cx q[1],q[7];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[1];
cx q[11],q[1];
rz(-pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[1];
cx q[1],q[3];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[11],q[14];
rz(1.2388685735269853) q[14];
cx q[11],q[14];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/4) q[11];
rz(0) q[11];
sx q[11];
rz(8.212140542390298) q[11];
sx q[11];
rz(3*pi) q[11];
rz(0) q[11];
sx q[11];
rz(5.7649267349731925) q[11];
sx q[11];
rz(3*pi) q[11];
rz(pi/2) q[11];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[3],q[1];
rz(-3.5881685855092007) q[1];
rz(pi/2) q[1];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[12];
cx q[12],q[3];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/4) q[12];
rz(3.5347482503890353) q[3];
rz(pi/2) q[3];
rz(pi/2) q[7];
rz(pi) q[7];
x q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[5];
rz(1.2895013558422341) q[5];
cx q[7],q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[14],q[5];
rz(5.225922885795281) q[5];
cx q[14],q[5];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(1.2475882704507955) q[7];
rz(5.7914056601316455) q[7];
rz(pi) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[17],q[9];
rz(6.053920712934922) q[9];
cx q[17],q[9];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(2.4438801033714537) q[17];
cx q[17],q[2];
rz(-2.4438801033714537) q[2];
cx q[17],q[2];
cx q[17],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
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
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[17],q[10];
cx q[10],q[17];
cx q[17],q[10];
rz(-pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[1];
rz(0) q[1];
sx q[1];
rz(0.3896940712033454) q[1];
sx q[1];
rz(3*pi) q[1];
rz(0) q[10];
sx q[10];
rz(5.89349123597624) q[10];
sx q[10];
rz(3*pi) q[10];
cx q[10],q[1];
rz(-pi/2) q[1];
rz(3.5881685855092007) q[1];
rz(-pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(4.483368248414344) q[10];
sx q[10];
rz(5*pi/2) q[10];
rz(0.7540090543825286) q[10];
rz(2.4597334620860414) q[10];
cx q[14],q[1];
cx q[1],q[14];
rz(1.1701333425853437) q[1];
sx q[14];
rz(1.9786575321414828) q[17];
rz(-pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[17],q[3];
rz(0) q[17];
sx q[17];
rz(1.3157155967157268) q[17];
sx q[17];
rz(3*pi) q[17];
rz(2.4438801033714537) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[2],q[4];
rz(0) q[3];
sx q[3];
rz(1.3157155967157268) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[17],q[3];
rz(-pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(-1.8573422906491084) q[17];
cx q[10],q[17];
rz(-2.4597334620860414) q[17];
sx q[17];
rz(2.6962028101257642) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[10],q[17];
rz(-pi/2) q[10];
rz(0) q[17];
sx q[17];
rz(3.586982497053822) q[17];
sx q[17];
rz(13.741853713504529) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(-pi/2) q[3];
rz(-3.5347482503890353) q[3];
rz(1.985079613052446) q[3];
sx q[3];
rz(7.156355333544372) q[3];
rz(-0.011517135367933307) q[3];
rz(2.059009883122796) q[4];
cx q[2],q[4];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/4) q[2];
cx q[2],q[13];
rz(-pi/4) q[13];
cx q[2],q[13];
rz(pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(2.5597373089741273) q[13];
rz(1.2704235346923642) q[2];
cx q[13],q[2];
cx q[2],q[13];
cx q[13],q[2];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(1.6307126689207867) q[13];
rz(-pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-pi/2) q[4];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/4) q[9];
cx q[9],q[15];
rz(-pi/4) q[15];
cx q[9],q[15];
rz(pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi) q[15];
x q[15];
rz(2.144405799155368) q[15];
sx q[15];
rz(5.071193124458432) q[15];
sx q[15];
rz(9.650303801902128) q[15];
rz(4.116652162024409) q[15];
cx q[15],q[4];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[5];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(1.360058091646589) q[4];
rz(3.9714820771080332) q[4];
cx q[5],q[15];
rz(-2.241297444388637) q[15];
cx q[4],q[15];
rz(-3.9714820771080332) q[15];
sx q[15];
rz(1.3206283392534366) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[4],q[15];
rz(0) q[15];
sx q[15];
rz(4.962556967926149) q[15];
sx q[15];
rz(15.63755748226605) q[15];
cx q[15],q[14];
cx q[14],q[15];
cx q[15],q[14];
rz(4.253170288824763) q[14];
rz(4.519065536453484) q[14];
rz(2.85601688200865) q[4];
cx q[4],q[3];
rz(-2.85601688200865) q[3];
sx q[3];
rz(1.0290534619007898) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[4],q[3];
rz(0) q[3];
sx q[3];
rz(5.254131845278796) q[3];
sx q[3];
rz(12.292311978145962) q[3];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[3],q[4];
rz(5.9150733162551745) q[4];
cx q[3],q[4];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(4.658540748150047) q[4];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-3.1750848170640937) q[5];
rz(pi/2) q[5];
rz(pi) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[16],q[9];
rz(4.114666093130178) q[9];
cx q[16],q[9];
rz(pi/4) q[16];
cx q[16],q[8];
rz(-pi/4) q[8];
cx q[16],q[8];
rz(0.5801601133898986) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[12],q[16];
rz(-pi/4) q[16];
cx q[12],q[16];
rz(0.6791327088450395) q[12];
rz(pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(4.24693612605733) q[16];
sx q[16];
rz(4.438854807968997) q[16];
sx q[16];
rz(12.37415034533236) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(1.6668093321586046) q[16];
cx q[6],q[12];
rz(-0.6791327088450395) q[12];
cx q[6],q[12];
cx q[12],q[16];
rz(-1.6668093321586046) q[16];
cx q[12],q[16];
rz(pi) q[12];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(0.7968293271292455) q[16];
cx q[14],q[16];
rz(-4.519065536453484) q[16];
sx q[16];
rz(0.24219746216512394) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[14],q[16];
rz(pi/2) q[14];
rz(0) q[16];
sx q[16];
rz(6.040987845014462) q[16];
sx q[16];
rz(13.147014170093618) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/4) q[16];
cx q[6],q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(-pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-0.18647560118432072) q[8];
cx q[7],q[8];
rz(-5.7914056601316455) q[8];
sx q[8];
rz(2.9897301872488864) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[7],q[8];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(0) q[8];
sx q[8];
rz(3.2934551199307) q[8];
sx q[8];
rz(15.402659222085346) q[8];
rz(pi/4) q[8];
cx q[8],q[7];
rz(-pi/4) q[7];
cx q[8],q[7];
rz(pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[5];
rz(0) q[5];
sx q[5];
rz(0.5767220333810674) q[5];
sx q[5];
rz(3*pi) q[5];
rz(0) q[7];
sx q[7];
rz(5.706463273798519) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[7],q[5];
rz(-pi/2) q[5];
rz(3.1750848170640937) q[5];
cx q[5],q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[5],q[10];
cx q[10],q[5];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
x q[10];
rz(1.1476161008656227) q[10];
sx q[10];
rz(2.000683435968156) q[10];
rz(-0.8516398545625035) q[5];
sx q[5];
rz(5.6826773254042156) q[5];
sx q[5];
rz(10.276417815331882) q[5];
rz(-pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(0.9502680297653523) q[7];
rz(2.803192017012535) q[7];
cx q[7],q[13];
rz(-2.803192017012535) q[13];
sx q[13];
rz(0.10116561064907659) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[7],q[13];
rz(0) q[13];
sx q[13];
rz(6.18201969653051) q[13];
sx q[13];
rz(10.597257308861128) q[13];
rz(0) q[13];
sx q[13];
rz(4.887612058987638) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[15],q[13];
rz(0) q[13];
sx q[13];
rz(1.3955732481919485) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[15],q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-0.2477760040468553) q[15];
cx q[4],q[15];
rz(-4.658540748150047) q[15];
sx q[15];
rz(0.955417250537419) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[4],q[15];
rz(0) q[15];
sx q[15];
rz(5.327768056642167) q[15];
sx q[15];
rz(14.331094712966282) q[15];
rz(2.072099354681482) q[15];
sx q[15];
rz(3.2590798420815124) q[15];
sx q[15];
rz(12.41571473075684) q[15];
rz(-pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[6];
rz(4.9055106952566305) q[6];
cx q[7],q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[6],q[14];
cx q[14],q[6];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
x q[14];
rz(-pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[6],q[4];
rz(2.4922083019366923) q[4];
cx q[6],q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
cx q[7],q[11];
rz(-0.25937001053647807) q[11];
rz(pi/2) q[11];
cx q[5],q[11];
rz(0) q[11];
sx q[11];
rz(1.5595904391935567) q[11];
sx q[11];
rz(3*pi) q[11];
rz(0) q[5];
sx q[5];
rz(4.72359486798603) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[5],q[11];
rz(-pi/2) q[11];
rz(0.25937001053647807) q[11];
rz(-pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(1.2684613867667744) q[7];
sx q[7];
rz(5.391218852816754) q[7];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/4) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[0];
cx q[0],q[9];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(8.946873930835213) q[0];
sx q[0];
rz(5*pi/2) q[0];
rz(-5.36949629095738) q[0];
rz(pi/2) q[0];
cx q[2],q[0];
rz(0) q[0];
sx q[0];
rz(2.8880822999796854) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[2];
sx q[2];
rz(3.395103007199901) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[2],q[0];
rz(-pi/2) q[0];
rz(5.36949629095738) q[0];
rz(-pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(4.798679032107321) q[2];
rz(pi/2) q[2];
cx q[13],q[2];
cx q[2],q[13];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(1.2987217188471498) q[2];
rz(pi/2) q[9];
rz(4.472493420687817) q[9];
sx q[9];
rz(8.683930261529943) q[9];
sx q[9];
rz(11.549667898626657) q[9];
cx q[9],q[8];
cx q[0],q[9];
rz(-pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-pi/2) q[8];
rz(2.105706267383829) q[8];
cx q[8],q[1];
rz(-2.105706267383829) q[1];
cx q[8],q[1];
rz(2.105706267383829) q[1];
rz(1.2157011146229983) q[1];
sx q[1];
rz(8.323676821963314) q[1];
sx q[1];
rz(8.209076846146381) q[1];
rz(4.92034734179885) q[1];
sx q[1];
rz(7.7859491597094435) q[1];
sx q[1];
rz(11.165292259949316) q[1];
rz(-pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(5.918003956538338) q[8];
cx q[8],q[16];
rz(-pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-pi/2) q[16];
rz(4.60223736891995) q[16];
sx q[16];
rz(7.285173096593212) q[16];
sx q[16];
rz(10.738453201444319) q[16];
rz(3.121247636396542) q[8];
cx q[13],q[8];
rz(-3.121247636396542) q[8];
cx q[13],q[8];
rz(2.3286425296185196) q[9];
cx q[0],q[9];
rz(pi) q[0];
x q[0];
rz(0.9164281128415134) q[0];
rz(pi/2) q[0];
rz(2.594772510740795) q[9];
cx q[17],q[9];
rz(-2.594772510740795) q[9];
cx q[17],q[9];
rz(2.9325146860321856) q[17];
cx q[12],q[17];
rz(-2.9325146860321856) q[17];
cx q[12],q[17];
rz(2.6867439928556465) q[12];
sx q[12];
rz(2.267943728888115) q[12];
sx q[17];
cx q[0],q[17];
x q[0];
cx q[9],q[3];
rz(4.081099268693586) q[3];
cx q[9],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(1.9734242639814092) q[3];
cx q[3],q[2];
rz(-1.9734242639814092) q[2];
sx q[2];
rz(0.1003961514050693) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[3],q[2];
rz(0) q[2];
sx q[2];
rz(6.1827891557745165) q[2];
sx q[2];
rz(10.099480505903639) q[2];
rz(5.76938148663405) q[9];
rz(pi/2) q[9];
cx q[1],q[9];
rz(0) q[1];
sx q[1];
rz(2.4199296609475187) q[1];
sx q[1];
rz(3*pi) q[1];
rz(0) q[9];
sx q[9];
rz(2.4199296609475187) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[1],q[9];
rz(-pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(-pi/2) q[9];
rz(-5.76938148663405) q[9];
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
