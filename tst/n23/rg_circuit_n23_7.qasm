OPENQASM 2.0;
include "qelib1.inc";
qreg q[23];
creg c[23];
rz(5.46495816588225) q[0];
sx q[0];
rz(7.582519370630829) q[0];
sx q[0];
rz(10.802579905552905) q[0];
rz(pi/4) q[0];
x q[0];
rz(2.287651616875695) q[0];
rz(0) q[1];
sx q[1];
rz(3.762049057481389) q[1];
sx q[1];
rz(3*pi) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(1.0501366168605486) q[6];
rz(3.5335217207630087) q[8];
rz(3.3037856190803776) q[8];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi) q[10];
x q[10];
rz(pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(4.124769106025567) q[11];
sx q[11];
rz(5*pi/2) q[11];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[13],q[2];
rz(pi/4) q[13];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[13],q[2];
rz(-pi/4) q[2];
cx q[13],q[2];
rz(3.1013398296858807) q[13];
sx q[13];
rz(9.063265993087239) q[13];
sx q[13];
rz(14.385378840280993) q[13];
rz(-pi/2) q[13];
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
cx q[10],q[2];
rz(3.318734397180145) q[2];
cx q[10],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(0.7336000007162828) q[2];
x q[2];
rz(0.8781055939312215) q[2];
rz(0) q[14];
sx q[14];
rz(4.36996973313289) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[5],q[14];
rz(0) q[14];
sx q[14];
rz(1.9132155740466967) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[5],q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(3.97901411732663) q[15];
sx q[15];
rz(7.487998020678973) q[15];
sx q[15];
rz(10.298418263156194) q[15];
cx q[15],q[5];
rz(pi/4) q[15];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[15],q[5];
rz(-pi/4) q[5];
cx q[15],q[5];
rz(pi/2) q[15];
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
rz(0.10739240240372047) q[16];
rz(pi/2) q[16];
cx q[12],q[16];
rz(0) q[12];
sx q[12];
rz(0.20615261703313514) q[12];
sx q[12];
rz(3*pi) q[12];
rz(0) q[16];
sx q[16];
rz(0.20615261703313514) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[12],q[16];
rz(-pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(-0.14830695481918976) q[12];
rz(-pi/2) q[16];
rz(-0.10739240240372047) q[16];
rz(0.4606317298006341) q[16];
rz(pi/2) q[16];
rz(pi) q[17];
x q[17];
rz(3.0806591753289343) q[18];
cx q[18],q[4];
rz(-3.0806591753289343) q[4];
cx q[18],q[4];
rz(pi) q[18];
x q[18];
rz(-2.83993202974282) q[18];
sx q[18];
rz(3.809109583614371) q[18];
sx q[18];
rz(12.264709990512198) q[18];
rz(5.325295551510992) q[18];
rz(2.7860307854686583) q[18];
rz(3.0806591753289343) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/4) q[4];
cx q[17],q[4];
rz(pi) q[17];
x q[17];
rz(4.032487972108977) q[17];
rz(pi) q[17];
rz(-pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-pi/2) q[4];
cx q[1],q[4];
rz(0) q[1];
sx q[1];
rz(5.702657966612781) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[6],q[19];
rz(-1.0501366168605486) q[19];
cx q[6],q[19];
rz(1.0501366168605486) q[19];
rz(-pi/4) q[19];
sx q[19];
rz(0) q[19];
sx q[19];
rz(3.377021929919095) q[19];
sx q[19];
rz(3*pi) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(3.8428676270444675) q[6];
rz(pi/4) q[6];
cx q[6],q[5];
rz(-pi/4) q[5];
cx q[6],q[5];
rz(pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(0.2190327464024527) q[5];
cx q[16],q[5];
rz(-0.2190327464024527) q[5];
cx q[16],q[5];
cx q[2],q[16];
rz(-0.8781055939312215) q[16];
cx q[2],q[16];
rz(0.8781055939312215) q[16];
sx q[16];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/4) q[5];
rz(3.138542561066409) q[5];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/4) q[6];
cx q[13],q[6];
rz(-2.417038808048252) q[13];
sx q[13];
rz(5.093569431972447) q[13];
sx q[13];
rz(11.841816768817631) q[13];
rz(0.9240346945340652) q[13];
rz(-pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(-pi/2) q[6];
cx q[20],q[3];
rz(0.24922127815554587) q[3];
cx q[20],q[3];
sx q[20];
id q[20];
rz(1.8958512690273177) q[20];
rz(2.798127787924007) q[20];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(0.820535671991319) q[3];
rz(4.040587262985182) q[3];
cx q[3],q[12];
rz(-4.040587262985182) q[12];
sx q[12];
rz(0.2835611362634558) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[3],q[12];
rz(0) q[12];
sx q[12];
rz(5.99962417091613) q[12];
sx q[12];
rz(13.61367217857375) q[12];
rz(-0.445149685896138) q[12];
rz(5.218728903249569) q[3];
rz(1.861137730662985) q[3];
cx q[3],q[12];
rz(-1.861137730662985) q[12];
sx q[12];
rz(0.47805008190675435) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[3],q[12];
rz(0) q[12];
sx q[12];
rz(5.805135225272831) q[12];
sx q[12];
rz(11.731065377328502) q[12];
rz(pi/2) q[12];
rz(-pi/2) q[3];
cx q[15],q[3];
rz(pi/2) q[3];
rz(0.12861555106465916) q[21];
cx q[8],q[21];
rz(-3.3037856190803776) q[21];
sx q[21];
rz(1.3351913029356128) q[21];
sx q[21];
rz(3*pi) q[21];
cx q[8],q[21];
rz(0) q[21];
sx q[21];
rz(4.947994004243974) q[21];
sx q[21];
rz(12.599948028785098) q[21];
rz(3.983443056951704) q[21];
sx q[21];
rz(5.990734312382464) q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/4) q[21];
cx q[11],q[21];
rz(0.36059207183129893) q[11];
cx q[20],q[11];
rz(-2.798127787924007) q[11];
sx q[11];
rz(1.661978481445803) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[20],q[11];
rz(0) q[11];
sx q[11];
rz(4.6212068257337835) q[11];
sx q[11];
rz(11.862313676862087) q[11];
rz(pi) q[11];
rz(5.116377612336214) q[11];
rz(-pi/2) q[20];
rz(-pi/4) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(-pi/2) q[21];
cx q[0],q[21];
rz(-2.287651616875695) q[21];
cx q[0],q[21];
rz(3.424672015646391) q[0];
rz(pi/2) q[0];
rz(2.287651616875695) q[21];
cx q[21],q[12];
cx q[12],q[21];
sx q[12];
cx q[3],q[20];
cx q[17],q[3];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[20],q[2];
rz(5.957835907687351) q[2];
cx q[20],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[16],q[20];
rz(pi/4) q[16];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[16],q[20];
rz(-pi/4) q[20];
cx q[16],q[20];
rz(pi/2) q[16];
rz(pi/4) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(-pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[3],q[17];
cx q[17],q[3];
rz(3.1002426876519085) q[17];
rz(-pi/2) q[3];
rz(pi) q[3];
x q[3];
rz(0) q[3];
sx q[3];
rz(5.324473187206385) q[3];
sx q[3];
rz(3*pi) q[3];
rz(5.063676575506969) q[3];
rz(pi/2) q[8];
cx q[14],q[8];
cx q[8],q[14];
rz(pi/2) q[14];
sx q[14];
rz(7.15842336427544) q[14];
sx q[14];
rz(5*pi/2) q[14];
sx q[14];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi) q[8];
cx q[8],q[1];
rz(0) q[1];
sx q[1];
rz(0.5805273405668054) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[8],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[1],q[19];
rz(3.062431210474517) q[19];
cx q[1],q[19];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
cx q[1],q[12];
x q[1];
rz(-0.12253041942960619) q[1];
rz(pi/4) q[12];
cx q[17],q[1];
rz(-3.1002426876519085) q[1];
sx q[1];
rz(2.0899825056640395) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[17],q[1];
rz(0) q[1];
sx q[1];
rz(4.193202801515547) q[1];
sx q[1];
rz(12.647551067850895) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(0) q[19];
sx q[19];
rz(5.284103213627212) q[19];
sx q[19];
rz(3*pi) q[19];
cx q[21],q[19];
rz(0) q[19];
sx q[19];
rz(0.9990820935523739) q[19];
sx q[19];
rz(3*pi) q[19];
cx q[21],q[19];
rz(-1.7283533759438068) q[19];
rz(pi/2) q[19];
rz(-pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[8],q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(1.3055678727390705) q[8];
sx q[8];
rz(6.305341062449558) q[8];
sx q[8];
rz(10.01870093324142) q[8];
rz(0) q[8];
sx q[8];
rz(3.2092102581321833) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[8],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(2.2064734256184115) q[8];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(1.3419494622155845) q[22];
cx q[7],q[22];
rz(-1.3419494622155845) q[22];
cx q[7],q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(5.887865841996748) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[4],q[22];
cx q[22],q[4];
cx q[14],q[22];
cx q[22],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
sx q[22];
cx q[11],q[22];
rz(0.721251216964464) q[22];
cx q[11],q[22];
rz(pi) q[22];
rz(pi/2) q[7];
cx q[9],q[7];
cx q[7],q[9];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/4) q[7];
cx q[10],q[7];
rz(-2.4730913354837325) q[10];
cx q[18],q[10];
rz(-2.7860307854686583) q[10];
sx q[10];
rz(1.9119958504636472) q[10];
sx q[10];
rz(3*pi) q[10];
cx q[18],q[10];
rz(0) q[10];
sx q[10];
rz(4.371189456715939) q[10];
sx q[10];
rz(14.68390008172177) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/4) q[10];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[18],q[0];
rz(0) q[0];
sx q[0];
rz(0.3879065001702604) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[18];
sx q[18];
rz(0.3879065001702604) q[18];
sx q[18];
rz(3*pi) q[18];
cx q[18],q[0];
rz(-pi/2) q[0];
rz(-3.424672015646391) q[0];
rz(0.6164954666952208) q[0];
rz(-pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
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
rz(5.8509387654359095) q[14];
rz(pi/2) q[14];
sx q[14];
rz(8.462284417562262) q[14];
sx q[14];
rz(5*pi/2) q[14];
rz(0) q[14];
sx q[14];
rz(3.4121761374858575) q[14];
sx q[14];
rz(3*pi) q[14];
rz(pi/2) q[14];
rz(-pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[18],q[21];
rz(0.39531835385703046) q[21];
cx q[18],q[21];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[18],q[16];
cx q[16],q[18];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/4) q[16];
cx q[16],q[1];
rz(-pi/4) q[1];
cx q[16],q[1];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
rz(3.442836016059473) q[21];
rz(-1.189420676851427) q[21];
cx q[4],q[10];
rz(-pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-pi/2) q[10];
cx q[10],q[0];
rz(-0.6164954666952208) q[0];
cx q[10],q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[2];
rz(0.8641448910028217) q[10];
sx q[10];
rz(7.312115390183261) q[10];
sx q[10];
rz(14.111455306436234) q[10];
rz(-0.17298581296430623) q[10];
cx q[2],q[0];
rz(1.9561588307830664) q[0];
rz(0.18316612222071713) q[0];
cx q[0],q[10];
rz(-0.18316612222071713) q[10];
sx q[10];
rz(2.2934851956126803) q[10];
sx q[10];
rz(3*pi) q[10];
cx q[0],q[10];
rz(0) q[10];
sx q[10];
rz(3.989700111566906) q[10];
sx q[10];
rz(9.780929895954403) q[10];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-0.9006197624124362) q[2];
rz(pi/2) q[2];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[6],q[4];
rz(0.556139865930931) q[4];
cx q[6],q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[12],q[4];
rz(-pi/4) q[4];
cx q[12],q[4];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[17];
cx q[12],q[16];
cx q[16],q[12];
cx q[12],q[16];
rz(pi/2) q[12];
sx q[12];
rz(7.824370329826538) q[12];
sx q[12];
rz(5*pi/2) q[12];
rz(1.9254892619463206) q[12];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(3.045106686910391) q[17];
rz(pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(0) q[4];
sx q[4];
rz(6.586992805787743) q[4];
sx q[4];
rz(3*pi) q[4];
rz(0) q[4];
sx q[4];
rz(5.368508621125847) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[22],q[4];
rz(0) q[4];
sx q[4];
rz(0.9146766860537392) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[22],q[4];
rz(5.3838451405384875) q[22];
cx q[22],q[16];
rz(2.6036274125159995) q[16];
cx q[22],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(4.028530750330074) q[16];
cx q[16],q[12];
rz(-4.028530750330074) q[12];
sx q[12];
rz(0.05115187084455508) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[16],q[12];
rz(0) q[12];
sx q[12];
rz(6.232033436335032) q[12];
sx q[12];
rz(11.527819449153132) q[12];
rz(-pi/4) q[12];
rz(pi/4) q[12];
rz(2.788166853661963) q[12];
rz(2.8956418786034277) q[12];
rz(-pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(5.221492929972808) q[22];
rz(-pi/4) q[22];
rz(2.7169543830871703) q[22];
rz(-pi/2) q[4];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(2.4053491134397573) q[6];
rz(-pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[6],q[2];
rz(0) q[2];
sx q[2];
rz(3.1392551012788545) q[2];
sx q[2];
rz(3*pi) q[2];
rz(0) q[6];
sx q[6];
rz(3.1439302059007317) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[6],q[2];
rz(-pi/2) q[2];
rz(0.9006197624124362) q[2];
rz(-pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(-pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi/2) q[7];
x q[7];
rz(0) q[7];
sx q[7];
rz(8.963902857573718) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[5],q[7];
rz(-3.138542561066409) q[7];
cx q[5],q[7];
rz(0) q[5];
sx q[5];
rz(3.4032551308354524) q[5];
sx q[5];
rz(3*pi) q[5];
rz(3.138542561066409) q[7];
cx q[7],q[5];
rz(0) q[5];
sx q[5];
rz(2.879930176344134) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[7],q[5];
rz(1.9254450966726306) q[5];
rz(3.68169893349319) q[7];
sx q[7];
rz(1.9462996740957283) q[7];
rz(pi/4) q[7];
cx q[7],q[4];
rz(pi/2) q[4];
rz(pi) q[4];
x q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[8],q[2];
rz(-2.2064734256184115) q[2];
cx q[8],q[2];
rz(2.2064734256184115) q[2];
rz(0.012045363745496931) q[8];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(4.392294323640355) q[9];
sx q[9];
rz(5*pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[15],q[9];
rz(2.816882157787675) q[15];
cx q[15],q[13];
rz(-2.816882157787675) q[13];
sx q[13];
rz(0.11919938275809194) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[15],q[13];
rz(0) q[13];
sx q[13];
rz(6.163985924421494) q[13];
sx q[13];
rz(11.317625424022989) q[13];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[19];
rz(0) q[13];
sx q[13];
rz(5.602719654986452) q[13];
sx q[13];
rz(3*pi) q[13];
rz(3.5275813673872958) q[15];
rz(pi/2) q[15];
rz(0) q[19];
sx q[19];
rz(0.680465652193134) q[19];
sx q[19];
rz(3*pi) q[19];
cx q[13],q[19];
rz(-pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(0) q[13];
sx q[13];
rz(4.928706788433499) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[11],q[13];
rz(0) q[13];
sx q[13];
rz(1.3544785187460875) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[11],q[13];
rz(-5.37111504495208) q[11];
rz(pi/2) q[11];
cx q[13],q[0];
rz(pi/2) q[0];
sx q[13];
cx q[14],q[13];
cx q[13],q[0];
rz(0.6861189257458314) q[0];
cx q[13],q[0];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
x q[14];
cx q[18],q[11];
rz(0) q[11];
sx q[11];
rz(2.544029507176372) q[11];
sx q[11];
rz(3*pi) q[11];
rz(0) q[18];
sx q[18];
rz(3.739155800003214) q[18];
sx q[18];
rz(3*pi) q[18];
cx q[18],q[11];
rz(-pi/2) q[11];
rz(5.37111504495208) q[11];
rz(0.4875410267622534) q[11];
sx q[11];
rz(9.302203988698922) q[11];
sx q[11];
rz(9.591941677178022) q[11];
rz(pi) q[11];
x q[11];
rz(pi) q[11];
x q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(2.6396571225113408) q[11];
rz(-pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(-pi/2) q[19];
rz(1.7283533759438068) q[19];
rz(1.2698998815104359) q[19];
sx q[19];
rz(7.016901971943332) q[19];
sx q[19];
rz(11.631852235025631) q[19];
rz(2.2160468362641113) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[18],q[19];
rz(pi/2) q[18];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(-pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/4) q[9];
rz(-pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[15];
rz(0) q[15];
sx q[15];
rz(0.18045128827039614) q[15];
sx q[15];
rz(3*pi) q[15];
rz(0) q[9];
sx q[9];
rz(0.18045128827039614) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[9],q[15];
rz(-pi/2) q[15];
rz(-3.5275813673872958) q[15];
cx q[5],q[15];
rz(-1.9254450966726306) q[15];
cx q[5],q[15];
rz(1.9254450966726306) q[15];
rz(2.329516470114755) q[15];
rz(1.468610773181597) q[15];
cx q[15],q[21];
rz(-1.468610773181597) q[21];
sx q[21];
rz(2.5197412553967413) q[21];
sx q[21];
rz(3*pi) q[21];
cx q[15],q[21];
id q[15];
sx q[15];
rz(0) q[21];
sx q[21];
rz(3.763444051782845) q[21];
sx q[21];
rz(12.082809410802403) q[21];
cx q[5],q[10];
rz(1.2659061136331033) q[10];
cx q[5],q[10];
rz(2.615660139609233) q[10];
cx q[10],q[2];
cx q[17],q[5];
rz(-2.615660139609233) q[2];
cx q[10],q[2];
sx q[10];
cx q[18],q[10];
cx q[10],q[4];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-pi/2) q[10];
x q[18];
rz(0.6103675796914786) q[18];
rz(2.615660139609233) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[14],q[2];
rz(pi/4) q[14];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[14],q[2];
rz(-pi/4) q[2];
cx q[14],q[2];
rz(0.37792873653790027) q[14];
rz(pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-3.045106686910391) q[5];
cx q[17],q[5];
rz(pi/2) q[17];
rz(3.045106686910391) q[5];
cx q[5],q[7];
cx q[7],q[5];
cx q[5],q[7];
rz(pi/2) q[5];
cx q[2],q[5];
cx q[5],q[2];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-2.218226764243702) q[7];
rz(pi/2) q[7];
cx q[13],q[7];
rz(0) q[13];
sx q[13];
rz(4.4498850431733015) q[13];
sx q[13];
rz(3*pi) q[13];
rz(0) q[7];
sx q[7];
rz(1.833300264006285) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[13],q[7];
rz(-pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(-pi/2) q[7];
rz(2.218226764243702) q[7];
rz(pi/2) q[7];
cx q[22],q[7];
rz(-2.7169543830871703) q[7];
cx q[22],q[7];
sx q[22];
rz(2.7169543830871703) q[7];
rz(0.9117804922674887) q[7];
sx q[7];
rz(4.06194983410013) q[7];
sx q[7];
rz(9.505134698583687) q[7];
sx q[7];
cx q[8],q[21];
rz(-0.012045363745496931) q[21];
cx q[8],q[21];
rz(0.012045363745496931) q[21];
rz(-pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[21],q[19];
rz(2.42907849639323) q[19];
cx q[21],q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
rz(-pi/4) q[19];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/4) q[21];
rz(1.909971557923274) q[8];
cx q[8],q[3];
rz(-1.909971557923274) q[3];
cx q[8],q[3];
rz(1.909971557923274) q[3];
cx q[3],q[21];
rz(-pi/4) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(-pi/2) q[21];
rz(pi/2) q[21];
sx q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[5],q[21];
cx q[21],q[5];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/4) q[21];
cx q[13],q[21];
rz(2.6336800645562515) q[13];
rz(-pi/2) q[13];
rz(-pi/4) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(-pi/2) q[21];
rz(0.938191308192058) q[21];
rz(3.6734786562675055) q[21];
rz(-pi/2) q[5];
rz(pi/2) q[8];
rz(-pi/2) q[8];
cx q[2],q[8];
rz(pi/2) q[8];
rz(3.609211748126826) q[8];
sx q[8];
rz(5.7280873874462355) q[8];
sx q[8];
rz(14.927512915413612) q[8];
rz(pi/4) q[8];
rz(-pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(-pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[20],q[9];
rz(1.122156173257271) q[9];
cx q[20],q[9];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
rz(-4.68569425802224) q[20];
sx q[20];
rz(5.55472029339523) q[20];
sx q[20];
rz(14.11047221879162) q[20];
rz(5.902390530467993) q[20];
sx q[20];
rz(5.591685834070584) q[20];
sx q[20];
rz(12.688328671266348) q[20];
cx q[1],q[20];
cx q[20],q[1];
cx q[1],q[20];
cx q[18],q[1];
rz(-0.6103675796914786) q[1];
cx q[18],q[1];
rz(0.6103675796914786) q[1];
rz(-pi/2) q[1];
rz(-pi/4) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(0.6549690193817529) q[18];
rz(-3.032555878774355) q[20];
sx q[20];
rz(3.952821599862807) q[20];
sx q[20];
rz(12.457333839543734) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
cx q[4],q[1];
rz(pi/2) q[1];
rz(0.5928820890418551) q[1];
sx q[4];
cx q[20],q[4];
x q[20];
rz(0.30356143061887364) q[20];
sx q[20];
rz(5.344453101138038) q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(5.755620178143534) q[20];
sx q[20];
rz(5*pi/2) q[20];
rz(0.8836006978819837) q[20];
sx q[20];
rz(7.192273553450392) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[5],q[4];
cx q[4],q[5];
rz(3.0654931419742764) q[4];
cx q[4],q[13];
sx q[13];
rz(pi/2) q[4];
cx q[4],q[13];
x q[4];
rz(2.66077127194641) q[4];
rz(-5.3316374180196755) q[4];
rz(pi/2) q[4];
rz(-pi/2) q[5];
rz(1.404375620270216) q[5];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[6];
rz(3.338912379411448) q[6];
cx q[9],q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(-pi/2) q[6];
rz(pi/2) q[6];
cx q[6],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(1.5008023820661454) q[15];
cx q[0],q[15];
rz(-1.5008023820661454) q[15];
cx q[0],q[15];
cx q[0],q[11];
rz(-2.6396571225113408) q[11];
cx q[0],q[11];
cx q[0],q[18];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[11];
cx q[11],q[3];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(1.6103169545999494) q[15];
cx q[15],q[1];
rz(-0.5928820890418551) q[1];
cx q[15],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(2.7821525750121983) q[1];
sx q[1];
rz(5.048140918384193) q[1];
sx q[1];
rz(11.023476671412235) q[1];
rz(pi/2) q[1];
rz(2.9877747770582666) q[15];
rz(-0.6549690193817529) q[18];
cx q[0],q[18];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi/4) q[3];
cx q[11],q[3];
rz(3.1324853257027945) q[11];
cx q[11],q[15];
rz(-3.1324853257027945) q[15];
sx q[15];
rz(2.6363187726037625) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[11],q[15];
rz(pi/2) q[11];
sx q[11];
rz(4.047557963647201) q[11];
sx q[11];
rz(5*pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(0) q[15];
sx q[15];
rz(3.6468665345758238) q[15];
sx q[15];
rz(9.569488509413906) q[15];
rz(1.9275664493438525) q[15];
x q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/4) q[15];
rz(pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
cx q[3],q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(-pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
x q[3];
rz(0.60445837908353) q[3];
rz(-3.959238577294088) q[3];
sx q[3];
rz(6.932926484075203) q[3];
sx q[3];
rz(13.384016538063467) q[3];
cx q[3],q[15];
rz(-pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(-pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[3];
x q[6];
rz(-3.767709767381315) q[6];
sx q[6];
rz(6.69278485576113) q[6];
sx q[6];
rz(13.192487728150695) q[6];
rz(pi/2) q[6];
cx q[8],q[18];
rz(-pi/4) q[18];
cx q[8],q[18];
rz(pi/4) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-2.4779707685937686) q[18];
rz(pi/2) q[18];
rz(0.12219280237717622) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[17];
cx q[17],q[9];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
x q[17];
rz(1.7748101791710802) q[17];
rz(pi/2) q[17];
cx q[16],q[17];
rz(0) q[16];
sx q[16];
rz(1.0120610065828592) q[16];
sx q[16];
rz(3*pi) q[16];
rz(0) q[17];
sx q[17];
rz(1.0120610065828592) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[16],q[17];
rz(-pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(4.112636269832186) q[16];
sx q[16];
rz(4.685257959466422) q[16];
cx q[16],q[10];
rz(3.8137078314165387) q[10];
cx q[16],q[10];
rz(-pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[18];
rz(0) q[10];
sx q[10];
rz(5.322880070941839) q[10];
sx q[10];
rz(3*pi) q[10];
rz(-pi/2) q[17];
rz(-1.7748101791710802) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(0) q[18];
sx q[18];
rz(0.960305236237748) q[18];
sx q[18];
rz(3*pi) q[18];
cx q[10],q[18];
rz(-pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(-pi/2) q[18];
rz(2.4779707685937686) q[18];
rz(6.0832929546114105) q[18];
rz(pi/2) q[18];
rz(3.7553613032850492) q[9];
rz(1.9341509021514107) q[9];
cx q[9],q[14];
rz(-1.9341509021514107) q[14];
sx q[14];
rz(1.8726766599211568) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[9],q[14];
rz(0) q[14];
sx q[14];
rz(4.410508647258429) q[14];
sx q[14];
rz(10.98100012638289) q[14];
rz(1.933521485996034) q[14];
cx q[19],q[14];
rz(-1.933521485996034) q[14];
cx q[19],q[14];
rz(-pi/4) q[19];
rz(pi/4) q[19];
rz(2.3365814856952163) q[19];
cx q[2],q[14];
rz(1.590477014267862) q[14];
rz(-0.8767942034602061) q[2];
cx q[12],q[2];
rz(-2.8956418786034277) q[2];
sx q[2];
rz(1.0741595954250651) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[12],q[2];
rz(0) q[2];
sx q[2];
rz(5.209025711754521) q[2];
sx q[2];
rz(13.197214042833014) q[2];
rz(4.409786451311485) q[2];
rz(1.2250147491909935) q[2];
cx q[2],q[8];
rz(-1.2250147491909935) q[8];
sx q[8];
rz(2.942489485341073) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[2],q[8];
sx q[2];
rz(2.1564450172771434) q[2];
rz(0) q[8];
sx q[8];
rz(3.3406958218385134) q[8];
sx q[8];
rz(10.527599907583197) q[8];
rz(2.427299674245746) q[8];
rz(pi/2) q[8];
cx q[22],q[8];
rz(0) q[22];
sx q[22];
rz(2.2710288276854937) q[22];
sx q[22];
rz(3*pi) q[22];
rz(0) q[8];
sx q[8];
rz(2.2710288276854937) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[22],q[8];
rz(-pi/2) q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/4) q[22];
rz(-pi/2) q[8];
rz(-2.427299674245746) q[8];
sx q[8];
cx q[18],q[8];
x q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
sx q[9];
cx q[6],q[9];
x q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[14],q[6];
rz(-1.590477014267862) q[6];
cx q[14],q[6];
cx q[19],q[14];
rz(-2.3365814856952163) q[14];
cx q[19],q[14];
rz(2.3365814856952163) q[14];
id q[14];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(2.1316012111260925) q[19];
rz(1.590477014267862) q[6];
rz(0.526687835352762) q[6];
cx q[6],q[12];
rz(-0.526687835352762) q[12];
cx q[6],q[12];
cx q[10],q[6];
rz(0.526687835352762) q[12];
cx q[5],q[12];
rz(-1.404375620270216) q[12];
cx q[5],q[12];
rz(1.404375620270216) q[12];
rz(3.003838652320094) q[12];
cx q[12],q[14];
rz(-3.003838652320094) q[14];
cx q[12],q[14];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[15];
rz(3.003838652320094) q[14];
sx q[14];
cx q[15],q[12];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(0) q[15];
sx q[15];
rz(4.8791992965093485) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[3],q[14];
x q[3];
rz(1.212233302278996) q[5];
sx q[5];
rz(5.445995675155245) q[5];
rz(5.9137746642876) q[5];
rz(4.190157984013889) q[5];
cx q[6],q[10];
cx q[10],q[6];
cx q[6],q[10];
cx q[10],q[6];
id q[10];
rz(5.8905241815563185) q[10];
cx q[9],q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/4) q[9];
cx q[9],q[17];
rz(-pi/4) q[17];
cx q[9],q[17];
rz(pi/4) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(-1.6916420367612326) q[17];
rz(pi/2) q[17];
cx q[0],q[17];
rz(0) q[0];
sx q[0];
rz(5.013066018417534) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[17];
sx q[17];
rz(1.2701192887620518) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[0],q[17];
rz(-pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(-pi/2) q[17];
rz(1.6916420367612326) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[16],q[17];
rz(pi/4) q[16];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[16],q[17];
rz(-pi/4) q[17];
cx q[16],q[17];
cx q[16],q[19];
rz(pi/4) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[17],q[1];
cx q[1],q[17];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(1.5154326004391032) q[1];
rz(pi/2) q[1];
cx q[11],q[1];
rz(0) q[1];
sx q[1];
rz(1.2891056278331883) q[1];
sx q[1];
rz(3*pi) q[1];
rz(0) q[11];
sx q[11];
rz(1.2891056278331883) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[11],q[1];
rz(-pi/2) q[1];
rz(-1.5154326004391032) q[1];
rz(2.8584712120736513) q[1];
rz(pi/2) q[1];
rz(-pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(-pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[17],q[22];
rz(-1.343172297506378) q[17];
rz(-2.1316012111260925) q[19];
cx q[16],q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(1.9815139840117029) q[19];
cx q[0],q[19];
rz(-1.9815139840117029) q[19];
cx q[0],q[19];
cx q[13],q[0];
cx q[0],q[13];
cx q[13],q[0];
rz(-1.197283692099042) q[0];
cx q[14],q[13];
cx q[13],q[14];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(1.081936346869693) q[19];
rz(-pi/4) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(-pi/2) q[22];
rz(-pi/2) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[22],q[11];
rz(3.7701281701130434) q[11];
cx q[22],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(0) q[11];
sx q[11];
rz(5.132651451633952) q[11];
sx q[11];
rz(3*pi) q[11];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
rz(0.1892603168015449) q[22];
cx q[5],q[17];
rz(-4.190157984013889) q[17];
sx q[17];
rz(2.9617229120833315) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[5],q[17];
rz(0) q[17];
sx q[17];
rz(3.3214623950962547) q[17];
sx q[17];
rz(14.958108242289647) q[17];
rz(2.7437038370514237) q[17];
cx q[17],q[18];
rz(-2.7437038370514237) q[18];
cx q[17],q[18];
rz(2.7437038370514237) q[18];
cx q[5],q[11];
rz(0) q[11];
sx q[11];
rz(1.1505338555456346) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[5],q[11];
cx q[8],q[19];
rz(-1.081936346869693) q[19];
cx q[8],q[19];
rz(-pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[19],q[4];
rz(0) q[19];
sx q[19];
rz(4.25221753564753) q[19];
sx q[19];
rz(3*pi) q[19];
rz(0) q[4];
sx q[4];
rz(2.030967771532057) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[19],q[4];
rz(-pi/2) q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
rz(-pi/2) q[4];
rz(5.3316374180196755) q[4];
rz(2.0955653599209993) q[8];
rz(4.0156492345208505) q[8];
cx q[8],q[0];
rz(-4.0156492345208505) q[0];
sx q[0];
rz(1.856804546684853) q[0];
sx q[0];
rz(3*pi) q[0];
cx q[8],q[0];
rz(0) q[0];
sx q[0];
rz(4.426380760494733) q[0];
sx q[0];
rz(14.637710887389272) q[0];
rz(-1.6012136283827336) q[9];
cx q[21],q[9];
rz(-3.6734786562675055) q[9];
sx q[9];
rz(2.3460635093969757) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[21],q[9];
rz(pi) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[21],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[20],q[16];
rz(5.96233173703872) q[16];
cx q[20],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(6.257128616259275) q[16];
sx q[16];
rz(8.110357792963153) q[16];
sx q[16];
rz(15.590245956149705) q[16];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
sx q[20];
rz(0) q[9];
sx q[9];
rz(3.9371217977826105) q[9];
sx q[9];
rz(14.699470245419619) q[9];
rz(pi/2) q[9];
cx q[9],q[7];
rz(pi) q[7];
x q[7];
rz(1.199641265628832) q[7];
sx q[7];
rz(7.420890129076066) q[7];
rz(-pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[1];
rz(0) q[1];
sx q[1];
rz(2.4671686149025467) q[1];
sx q[1];
rz(3*pi) q[1];
rz(0) q[7];
sx q[7];
rz(2.4671686149025467) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[7],q[1];
rz(-pi/2) q[1];
rz(-2.8584712120736513) q[1];
rz(-pi/2) q[1];
cx q[12],q[1];
rz(pi/2) q[1];
rz(-pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(0.9080085684657409) q[7];
x q[9];
rz(-3.563583824587597) q[9];
sx q[9];
rz(6.448249727519917) q[9];
sx q[9];
rz(12.988361785356975) q[9];
cx q[9],q[2];
rz(-2.1564450172771434) q[2];
cx q[9],q[2];
cx q[2],q[21];
cx q[21],q[2];
cx q[2],q[21];
cx q[2],q[15];
rz(0) q[15];
sx q[15];
rz(1.4039860106702375) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[2],q[15];
rz(4.810159294261271) q[21];
rz(-pi/2) q[9];
cx q[6],q[9];
rz(pi) q[6];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[3],q[9];
rz(4.988146506653504) q[9];
cx q[3],q[9];
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
measure q[20] -> c[20];
measure q[21] -> c[21];
measure q[22] -> c[22];