OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
rz(0) q[0];
sx q[0];
rz(5.434217679806052) q[0];
sx q[0];
rz(3*pi) q[0];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(0.5605632550915098) q[2];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
id q[4];
rz(2.938606320037828) q[4];
rz(0.9124761415633265) q[4];
rz(-pi/4) q[5];
rz(1.8422885717553579) q[5];
rz(2.6768429926250055) q[6];
rz(pi/4) q[6];
rz(-pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[1],q[8];
rz(1.5736775799149871) q[8];
cx q[1],q[8];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[1],q[5];
rz(-1.8422885717553579) q[5];
cx q[1],q[5];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[5];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(5.449003935816659) q[9];
rz(2.7269640763096907) q[9];
cx q[9],q[2];
rz(-2.7269640763096907) q[2];
sx q[2];
rz(1.5821999888280127) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[9],q[2];
rz(0) q[2];
sx q[2];
rz(4.700985318351574) q[2];
sx q[2];
rz(11.59117878198756) q[2];
rz(pi/2) q[2];
sx q[2];
cx q[8],q[2];
x q[8];
rz(0) q[8];
sx q[8];
rz(5.592838203649158) q[8];
sx q[8];
rz(3*pi) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/4) q[10];
rz(2.71280471367225) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[10],q[13];
rz(-pi/4) q[13];
cx q[10],q[13];
rz(pi) q[10];
rz(0.9584249383055139) q[10];
sx q[10];
rz(6.265518910222863) q[10];
sx q[10];
rz(14.237756250532577) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[14],q[0];
rz(0) q[0];
sx q[0];
rz(0.848967627373534) q[0];
sx q[0];
rz(3*pi) q[0];
cx q[14],q[0];
rz(0.8869129861942104) q[0];
sx q[0];
rz(1.8321835230465253) q[0];
rz(0.2137532027206217) q[0];
sx q[0];
rz(4.893105184248229) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(5.285547126663969) q[14];
id q[14];
rz(pi) q[14];
x q[14];
id q[14];
rz(pi/2) q[14];
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
rz(4.09740096662016) q[15];
sx q[15];
rz(1.6790074250512754) q[15];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(1.4599216035981457) q[7];
cx q[13],q[7];
rz(-1.4599216035981457) q[7];
cx q[13],q[7];
rz(4.437361828706462) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[15];
rz(3.017753583051212) q[15];
cx q[7],q[15];
rz(2.918623003975062) q[15];
rz(0.6610733200671768) q[16];
rz(2.488972833667335) q[16];
rz(pi/4) q[17];
cx q[17],q[3];
rz(-pi/4) q[3];
cx q[17],q[3];
rz(0.2531314052903483) q[17];
rz(pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/4) q[3];
rz(0) q[3];
sx q[3];
rz(7.4014127677299255) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[3],q[10];
rz(0.7488225913606611) q[10];
cx q[3],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-2.03700449022382) q[18];
cx q[16],q[18];
rz(-2.488972833667335) q[18];
sx q[18];
rz(1.9613639685109272) q[18];
sx q[18];
rz(3*pi) q[18];
cx q[16],q[18];
rz(0) q[18];
sx q[18];
rz(4.321821338668659) q[18];
sx q[18];
rz(13.950755284660534) q[18];
cx q[18],q[9];
rz(2.012997659831966) q[9];
cx q[18],q[9];
rz(-pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[6],q[18];
rz(4.66021646056651) q[18];
cx q[6],q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
cx q[18],q[8];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[2],q[6];
rz(-pi/2) q[2];
cx q[3],q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/4) q[2];
rz(-3.1893205809058953) q[3];
sx q[3];
rz(4.664183698323277) q[3];
sx q[3];
rz(12.614098541675276) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(2.4995605837974435) q[6];
sx q[6];
rz(4.732794488877117) q[6];
sx q[6];
rz(15.102936773230386) q[6];
rz(-pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(0) q[8];
sx q[8];
rz(0.6903471035304278) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[18],q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(4.624306988824723) q[9];
cx q[9],q[4];
rz(-4.624306988824723) q[4];
sx q[4];
rz(0.5689037664740826) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[9],q[4];
rz(0) q[4];
sx q[4];
rz(5.714281540705503) q[4];
sx q[4];
rz(13.136608808030775) q[4];
rz(0.6875810407617654) q[4];
cx q[9],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/4) q[9];
cx q[9],q[0];
rz(-pi/4) q[0];
cx q[9],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/4) q[0];
cx q[9],q[18];
rz(5.966024053138755) q[18];
cx q[9],q[18];
rz(2.2269769422146735) q[19];
cx q[19],q[12];
rz(-2.2269769422146735) q[12];
cx q[19],q[12];
rz(2.2269769422146735) q[12];
cx q[17],q[12];
rz(-0.2531314052903483) q[12];
cx q[17],q[12];
rz(0.2531314052903483) q[12];
rz(-pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[11];
rz(2.7303693454447266) q[11];
cx q[12],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(5.160370403595742) q[11];
sx q[11];
rz(5*pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
cx q[12],q[4];
sx q[17];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[16],q[19];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[1];
rz(2.1735571438260384) q[1];
cx q[16],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
cx q[13],q[1];
cx q[1],q[13];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(1.6750117033112177) q[1];
cx q[13],q[0];
rz(-pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/2) q[0];
rz(0.994217994719911) q[0];
rz(-pi/2) q[13];
rz(pi) q[13];
x q[13];
rz(-3.622202794445363) q[13];
rz(pi/2) q[13];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[15],q[16];
rz(-2.918623003975062) q[16];
cx q[15],q[16];
rz(4.301737045371712) q[15];
sx q[15];
rz(9.197438148988049) q[15];
sx q[15];
rz(11.651410622292547) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(2.918623003975062) q[16];
rz(1.3144276930540972) q[16];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(0) q[19];
sx q[19];
rz(7.604143521953749) q[19];
sx q[19];
rz(3*pi) q[19];
rz(pi/4) q[19];
rz(0) q[19];
sx q[19];
rz(5.1330808331638575) q[19];
sx q[19];
rz(3*pi) q[19];
rz(-0.6875810407617654) q[4];
cx q[12],q[4];
cx q[12],q[16];
rz(-1.3144276930540972) q[16];
cx q[12],q[16];
rz(1.3266329876767924) q[12];
rz(0.3303452833667327) q[12];
rz(-pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[5],q[17];
rz(-pi/2) q[17];
x q[5];
rz(0) q[5];
sx q[5];
rz(3.4351270825577744) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[5],q[19];
rz(0) q[19];
sx q[19];
rz(1.1501044740157291) q[19];
sx q[19];
rz(3*pi) q[19];
cx q[5],q[19];
rz(pi) q[19];
x q[19];
rz(pi/4) q[5];
cx q[5],q[10];
rz(-pi/4) q[10];
cx q[5],q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(0) q[5];
sx q[5];
rz(5.502533365163867) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[6],q[16];
rz(1.8497705469944583) q[16];
cx q[6],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(4.740268819437432) q[16];
sx q[16];
rz(5.772954581280748) q[16];
sx q[16];
rz(14.517412079303881) q[16];
sx q[16];
rz(0) q[16];
sx q[16];
rz(6.004556767100257) q[16];
sx q[16];
rz(3*pi) q[16];
rz(-pi/4) q[16];
rz(0.8978210502929049) q[16];
sx q[16];
rz(4.797210611024749) q[16];
sx q[16];
rz(14.958556488745895) q[16];
sx q[16];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(9.29833990462758) q[6];
sx q[6];
rz(5*pi/2) q[6];
rz(-pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[6],q[13];
rz(0) q[13];
sx q[13];
rz(2.6910964489618463) q[13];
sx q[13];
rz(3*pi) q[13];
rz(0) q[6];
sx q[6];
rz(3.59208885821774) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[6],q[13];
rz(-pi/2) q[13];
rz(3.622202794445363) q[13];
rz(pi) q[13];
rz(-pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi) q[6];
x q[6];
rz(-pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[7],q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[11],q[17];
rz(4.481071465141631) q[17];
cx q[11],q[17];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[11],q[2];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
sx q[17];
cx q[14],q[17];
x q[14];
rz(2.6528469301738995) q[14];
rz(1.4503195343226356) q[17];
rz(-pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-pi/2) q[2];
cx q[19],q[2];
rz(6.1656496997706505) q[2];
cx q[19],q[2];
rz(-pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[4];
rz(5.405712692291766) q[4];
cx q[7],q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(pi/4) q[4];
cx q[4],q[15];
rz(-pi/4) q[15];
cx q[4],q[15];
rz(pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(0) q[4];
sx q[4];
rz(5.3379827541044715) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[10],q[4];
rz(0) q[4];
sx q[4];
rz(0.9452025530751151) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[10],q[4];
cx q[4],q[12];
rz(4.983008582181373) q[4];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
cx q[8],q[1];
rz(-1.6750117033112177) q[1];
cx q[8],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(0.7439127270300708) q[1];
cx q[1],q[18];
rz(-0.7439127270300708) q[18];
cx q[1],q[18];
cx q[14],q[1];
rz(-2.6528469301738995) q[1];
cx q[14],q[1];
rz(2.6528469301738995) q[1];
rz(0) q[1];
sx q[1];
rz(8.06014800312471) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[12],q[1];
cx q[1],q[12];
cx q[12],q[1];
sx q[1];
rz(-0.11338422968412015) q[1];
rz(4.197235053032295) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(0) q[14];
sx q[14];
rz(3.335327026909456) q[14];
sx q[14];
rz(3*pi) q[14];
rz(0.7439127270300708) q[18];
cx q[18],q[3];
rz(3.0534684695691845) q[3];
cx q[18],q[3];
rz(-2.3876976680597726) q[18];
sx q[18];
rz(8.145731286290488) q[18];
sx q[18];
rz(11.812475628829151) q[18];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[14];
rz(0) q[14];
sx q[14];
rz(2.9478582802701303) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[3],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[18],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/4) q[18];
cx q[18],q[14];
rz(-pi/4) q[14];
cx q[18],q[14];
rz(pi/4) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(4.657159817563661) q[14];
rz(3.5116785735654035) q[14];
rz(-pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(1.1022609427557992) q[3];
cx q[4],q[3];
rz(-4.983008582181373) q[3];
sx q[3];
rz(0.1578527993509482) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[4],q[3];
rz(0) q[3];
sx q[3];
rz(6.125332507828638) q[3];
sx q[3];
rz(13.305525600194953) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/4) q[4];
cx q[4],q[3];
rz(-pi/4) q[3];
cx q[4],q[3];
rz(pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(1.2746186928868528) q[3];
rz(1.0058444757340068) q[3];
cx q[3],q[1];
rz(-1.0058444757340068) q[1];
sx q[1];
rz(0.8437854376159066) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[3],q[1];
rz(0) q[1];
sx q[1];
rz(5.43939986956368) q[1];
sx q[1];
rz(10.544006666187506) q[1];
rz(-pi/4) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[6],q[18];
rz(1.9321933130792674) q[18];
cx q[6],q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(-0.04484657004796927) q[8];
cx q[0],q[8];
rz(-0.994217994719911) q[8];
sx q[8];
rz(2.5868072925706302) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[0],q[8];
cx q[0],q[5];
rz(0) q[5];
sx q[5];
rz(0.7806519420157194) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[0],q[5];
cx q[19],q[0];
cx q[0],q[19];
id q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(2.0400797376553386) q[0];
cx q[13],q[0];
rz(-2.0400797376553386) q[0];
cx q[13],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/4) q[0];
cx q[1],q[0];
cx q[0],q[1];
cx q[1],q[0];
rz(-pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(2.225329480261648) q[13];
rz(3.443368411189174) q[13];
sx q[13];
rz(3.5741014812721743) q[13];
rz(pi) q[13];
rz(-2.7500479785797864) q[13];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(2.418256013606716) q[5];
cx q[2],q[5];
rz(-2.418256013606716) q[5];
cx q[2],q[5];
rz(-pi/2) q[2];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(0) q[5];
sx q[5];
rz(6.190935050656949) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[19],q[5];
rz(0) q[5];
sx q[5];
rz(0.09225025652263685) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[19],q[5];
rz(-1.3732685856636855) q[19];
cx q[14],q[19];
rz(-3.5116785735654035) q[19];
sx q[19];
rz(1.975912072558355) q[19];
sx q[19];
rz(3*pi) q[19];
cx q[14],q[19];
rz(1.3308510483863514) q[14];
rz(0) q[19];
sx q[19];
rz(4.307273234621231) q[19];
sx q[19];
rz(14.309725119998468) q[19];
rz(0.3719257200525985) q[19];
sx q[19];
rz(2.4924423702333565) q[19];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(0) q[8];
sx q[8];
rz(3.696378014608956) q[8];
sx q[8];
rz(10.46384252553726) q[8];
cx q[8],q[17];
rz(-1.4503195343226356) q[17];
cx q[8],q[17];
cx q[8],q[10];
rz(pi/2) q[10];
sx q[8];
cx q[10],q[8];
x q[10];
rz(2.997180489378427) q[10];
rz(1.4252539966877373) q[8];
cx q[9],q[7];
cx q[7],q[9];
cx q[7],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(2.2085959060304052) q[15];
cx q[15],q[17];
rz(-2.2085959060304052) q[17];
cx q[15],q[17];
cx q[15],q[2];
cx q[10],q[15];
rz(-2.997180489378427) q[15];
cx q[10],q[15];
rz(0) q[10];
sx q[10];
rz(8.159639506495973) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[10];
sx q[10];
rz(4.565373868537623) q[10];
sx q[10];
rz(3*pi) q[10];
rz(2.997180489378427) q[15];
rz(5.360093650057048) q[15];
rz(2.2085959060304052) q[17];
rz(pi) q[17];
cx q[17],q[8];
cx q[18],q[10];
rz(0) q[10];
sx q[10];
rz(1.7178114386419627) q[10];
sx q[10];
rz(3*pi) q[10];
cx q[18],q[10];
rz(pi) q[10];
x q[10];
id q[10];
rz(-1.6932420393926222) q[10];
sx q[10];
rz(3.1682124520875083) q[10];
sx q[10];
rz(11.118020000162002) q[10];
rz(4.037693546242303) q[10];
rz(pi/2) q[10];
rz(1.983936063204386) q[18];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/4) q[2];
cx q[2],q[12];
rz(-pi/4) q[12];
cx q[2],q[12];
rz(pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
sx q[12];
x q[12];
rz(-pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(2.6134714896695623) q[2];
rz(0.028611344125920812) q[7];
rz(-1.4252539966877373) q[8];
cx q[17],q[8];
cx q[6],q[17];
rz(pi) q[17];
x q[17];
rz(-pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[0],q[17];
rz(0.7854526188189864) q[17];
cx q[0],q[17];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(-pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
cx q[2],q[17];
rz(-2.6134714896695623) q[17];
cx q[2],q[17];
rz(2.6134714896695623) q[17];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[6],q[3];
cx q[3],q[6];
cx q[6],q[3];
rz(pi/2) q[3];
sx q[3];
rz(5.427620060088326) q[3];
sx q[3];
rz(5*pi/2) q[3];
rz(pi/2) q[6];
rz(5.655514654229877) q[8];
rz(pi) q[8];
x q[8];
rz(pi) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(0) q[9];
sx q[9];
rz(5.182706184381792) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[11],q[9];
rz(0) q[9];
sx q[9];
rz(1.1004791227977941) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[11],q[9];
rz(5.778305528464289) q[11];
cx q[11],q[7];
rz(-5.778305528464289) q[7];
sx q[7];
rz(1.0535007400827303) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[11],q[7];
rz(-pi/2) q[11];
rz(-pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(0) q[7];
sx q[7];
rz(5.2296845670968555) q[7];
sx q[7];
rz(15.174472145107746) q[7];
rz(pi) q[7];
x q[7];
rz(0.7387087498596226) q[7];
rz(pi/2) q[7];
cx q[11],q[7];
rz(0) q[11];
sx q[11];
rz(1.0949564973612542) q[11];
sx q[11];
rz(3*pi) q[11];
rz(0) q[7];
sx q[7];
rz(1.0949564973612542) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[11],q[7];
rz(-pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/2) q[7];
rz(-0.7387087498596226) q[7];
cx q[14],q[7];
rz(-1.3308510483863514) q[7];
cx q[14],q[7];
rz(1.9797618137278818) q[14];
rz(0.4138129203331349) q[14];
sx q[14];
rz(7.070557583031382) q[14];
sx q[14];
rz(9.010965040436245) q[14];
rz(1.3308510483863514) q[7];
cx q[7],q[15];
rz(4.148735965188594) q[15];
cx q[7],q[15];
rz(1.2942972850646222) q[15];
sx q[15];
rz(8.794649400536162) q[15];
sx q[15];
rz(11.827259516166635) q[15];
rz(0) q[15];
sx q[15];
rz(3.5275005009746265) q[15];
sx q[15];
rz(3*pi) q[15];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[4];
cx q[4],q[7];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[3],q[4];
cx q[4],q[3];
cx q[3],q[4];
rz(pi/2) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[0],q[7];
rz(2.2317131442187583) q[7];
cx q[0],q[7];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi) q[0];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(0.6893491744747252) q[7];
rz(-0.18802665172809796) q[9];
sx q[9];
rz(3.495506626407348) q[9];
rz(4.485869851647533) q[9];
rz(pi/4) q[9];
cx q[9],q[5];
rz(-pi/4) q[5];
cx q[9],q[5];
rz(pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
cx q[11],q[5];
cx q[5],q[11];
rz(pi) q[11];
x q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[11],q[6];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[19],q[5];
rz(5.476081730378883) q[5];
cx q[19],q[5];
rz(0) q[19];
sx q[19];
rz(5.39841321399978) q[19];
sx q[19];
rz(3*pi) q[19];
cx q[1],q[19];
rz(0) q[19];
sx q[19];
rz(0.884772093179806) q[19];
sx q[19];
rz(3*pi) q[19];
cx q[1],q[19];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[12],q[19];
rz(4.446478839449525) q[19];
cx q[12],q[19];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(0.7658760856430011) q[12];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
rz(-pi/2) q[5];
cx q[6],q[11];
cx q[11],q[15];
rz(0) q[15];
sx q[15];
rz(2.7556848062049597) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[11],q[15];
rz(pi/2) q[11];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(3.5779240506074688) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/4) q[6];
cx q[17],q[6];
rz(4.747847522098122) q[17];
sx q[17];
rz(3.5892916456717687) q[17];
sx q[17];
rz(14.24904264827554) q[17];
rz(4.617789565081248) q[17];
rz(pi/2) q[17];
rz(-pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(-pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[9];
sx q[9];
rz(6.451061170630144) q[9];
sx q[9];
rz(5*pi/2) q[9];
rz(pi/2) q[9];
cx q[9],q[16];
cx q[16],q[5];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[8];
rz(pi/2) q[5];
cx q[14],q[5];
cx q[5],q[14];
cx q[14],q[5];
cx q[14],q[7];
cx q[5],q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(2.4885832911717816) q[4];
cx q[3],q[4];
rz(-2.4885832911717816) q[4];
cx q[3],q[4];
rz(-pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(0.6633486723243596) q[4];
rz(pi/2) q[4];
cx q[3],q[4];
rz(0) q[3];
sx q[3];
rz(0.9607286217104423) q[3];
sx q[3];
rz(3*pi) q[3];
rz(0) q[4];
sx q[4];
rz(0.9607286217104423) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[3],q[4];
rz(-pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[4];
rz(-0.6633486723243596) q[4];
rz(2.147632201702706) q[5];
rz(-0.6893491744747252) q[7];
cx q[14],q[7];
rz(0.9323520700196576) q[14];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[0],q[7];
rz(4.04282516550818) q[7];
cx q[0],q[7];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(0.43341704895084426) q[8];
cx q[16],q[8];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-pi/4) q[16];
cx q[16],q[15];
rz(2.3383348747279005) q[15];
cx q[16],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[0],q[15];
rz(0.634536427306786) q[15];
cx q[0],q[15];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[12],q[8];
rz(-0.7658760856430011) q[8];
cx q[12],q[8];
x q[12];
cx q[12],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(5.680986314870152) q[16];
rz(0.7658760856430011) q[8];
cx q[8],q[14];
rz(-0.9323520700196576) q[14];
cx q[8],q[14];
rz(-pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[17];
rz(0) q[14];
sx q[14];
rz(0.9568506767949461) q[14];
sx q[14];
rz(3*pi) q[14];
rz(0) q[17];
sx q[17];
rz(0.9568506767949461) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[14],q[17];
rz(-pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(6.388634684911958) q[14];
sx q[14];
rz(5*pi/2) q[14];
rz(0.5250359626519979) q[14];
cx q[14],q[16];
rz(-0.5250359626519979) q[16];
cx q[14],q[16];
rz(0.5250359626519979) q[16];
rz(pi/4) q[16];
rz(-pi/2) q[17];
rz(-4.617789565081248) q[17];
rz(-pi/2) q[17];
rz(4.160054648224715) q[17];
rz(pi/2) q[17];
x q[9];
cx q[18],q[9];
rz(-1.983936063204386) q[9];
cx q[18],q[9];
cx q[18],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[18];
cx q[18],q[1];
rz(-pi/4) q[1];
cx q[18],q[1];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
sx q[1];
cx q[11],q[1];
x q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
x q[11];
rz(0.25415760471969806) q[11];
rz(pi) q[18];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[18],q[10];
rz(0) q[10];
sx q[10];
rz(1.343598904195317) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[18];
sx q[18];
rz(1.343598904195317) q[18];
sx q[18];
rz(3*pi) q[18];
cx q[18],q[10];
rz(-pi/2) q[10];
rz(-4.037693546242303) q[10];
rz(-pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
cx q[18],q[4];
rz(6.109035335681699) q[4];
cx q[18],q[4];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi/2) q[4];
cx q[7],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(0.1537720263961928) q[1];
rz(1.983936063204386) q[9];
rz(0.042260844985075374) q[9];
rz(2.9068406516545844) q[9];
cx q[9],q[13];
rz(-2.9068406516545844) q[13];
sx q[13];
rz(0.9823706765665352) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[9],q[13];
rz(0) q[13];
sx q[13];
rz(5.300814630613051) q[13];
sx q[13];
rz(15.08166659100375) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[2];
rz(4.857896553515001) q[2];
cx q[13],q[2];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(0.22598696799198373) q[13];
rz(6.0068576462524685) q[13];
cx q[13],q[11];
rz(-6.0068576462524685) q[11];
sx q[11];
rz(0.686530122201654) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[13],q[11];
rz(0) q[11];
sx q[11];
rz(5.596655184977932) q[11];
sx q[11];
rz(15.177478002302148) q[11];
rz(0.6968931149398581) q[11];
rz(pi/2) q[11];
cx q[11],q[15];
x q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(3.2228525358551368) q[13];
sx q[13];
rz(3.516762538187322) q[13];
cx q[13],q[7];
rz(pi/2) q[15];
sx q[15];
rz(7.813819392178372) q[15];
sx q[15];
rz(5*pi/2) q[15];
rz(1.349995820106056) q[15];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[7],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[18],q[13];
rz(1.5812846168862429) q[13];
cx q[18],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(2.589680822008818) q[13];
sx q[13];
rz(4.869738439956468) q[13];
sx q[13];
rz(13.999846029363479) q[13];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(1.6442795224444327) q[9];
cx q[9],q[19];
rz(-1.6442795224444327) q[19];
cx q[9],q[19];
rz(1.6442795224444327) q[19];
cx q[19],q[6];
rz(pi/4) q[19];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[19],q[6];
rz(-pi/4) q[6];
cx q[19],q[6];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(0.004420830832780275) q[19];
cx q[10],q[19];
rz(-0.004420830832780275) q[19];
cx q[10],q[19];
rz(-2.6465909416332094) q[10];
sx q[10];
rz(5.844756328926894) q[10];
sx q[10];
rz(12.07136890240259) q[10];
rz(-pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[17];
rz(0) q[10];
sx q[10];
rz(1.257941070900668) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[17];
sx q[17];
rz(1.257941070900668) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[10],q[17];
rz(-pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
cx q[14],q[10];
rz(2.95569560611599) q[10];
cx q[14],q[10];
rz(-pi/2) q[17];
rz(-4.160054648224715) q[17];
rz(5.026913788849272) q[17];
rz(1.4148151096774884) q[17];
cx q[17],q[1];
rz(-1.4148151096774884) q[1];
sx q[1];
rz(1.8248656304313127) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[17],q[1];
rz(0) q[1];
sx q[1];
rz(4.458319676748274) q[1];
sx q[1];
rz(10.685821044050675) q[1];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
id q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(1.8074700420372485) q[19];
rz(pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[5],q[6];
rz(5.130618067258685) q[6];
cx q[5],q[6];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
cx q[5],q[19];
rz(-1.8074700420372485) q[19];
cx q[5],q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(0) q[19];
sx q[19];
rz(9.363638561962272) q[19];
sx q[19];
rz(3*pi) q[19];
rz(-pi/2) q[5];
cx q[6],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[6],q[4];
rz(pi/2) q[4];
rz(4.18200089698817) q[4];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[2],q[9];
rz(pi/4) q[2];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[2],q[9];
rz(-pi/4) q[9];
cx q[2],q[9];
cx q[8],q[2];
rz(3.953941244899404) q[2];
cx q[8],q[2];
cx q[2],q[0];
cx q[0],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(0.9385894218838221) q[0];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[3],q[0];
rz(-0.9385894218838221) q[0];
cx q[3],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[18];
cx q[18],q[0];
cx q[0],q[18];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(-pi/2) q[8];
cx q[12],q[8];
cx q[12],q[2];
rz(pi/4) q[12];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[12],q[2];
rz(-pi/4) q[2];
cx q[12],q[2];
rz(-pi/4) q[12];
rz(pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[8];
rz(pi/4) q[8];
cx q[8],q[11];
rz(-pi/4) q[11];
cx q[8],q[11];
rz(pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[11],q[15];
rz(-1.349995820106056) q[15];
cx q[11],q[15];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[2],q[8];
rz(pi/4) q[2];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[2],q[8];
rz(-pi/4) q[8];
cx q[2],q[8];
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/4) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(1.7322406943203732) q[9];
sx q[9];
rz(3.8035055451842674) q[9];
rz(pi/4) q[9];
cx q[9],q[7];
rz(-pi/4) q[7];
cx q[9],q[7];
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
cx q[6],q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[16],q[9];
rz(-pi/4) q[9];
cx q[16],q[9];
rz(pi/4) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
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
