OPENQASM 2.0;
include "qelib1.inc";
qreg q[28];
creg c[28];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/4) q[1];
rz(pi/2) q[2];
rz(2.288394411247213) q[3];
rz(2.165140739377197) q[3];
rz(3.900447886791652) q[4];
rz(pi/2) q[4];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[1],q[6];
rz(-pi/4) q[6];
cx q[1],q[6];
rz(pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
sx q[7];
cx q[2],q[7];
x q[2];
rz(pi/2) q[2];
x q[8];
rz(5.521195133664942) q[8];
rz(4.754191190690358) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(0.476842257587859) q[9];
cx q[5],q[10];
rz(2.988360873084825) q[10];
cx q[5],q[10];
rz(0) q[10];
sx q[10];
rz(8.404084444247005) q[10];
sx q[10];
rz(3*pi) q[10];
rz(-2.5864077767639246) q[10];
sx q[10];
rz(6.313654184339531) q[10];
sx q[10];
rz(12.011185737533303) q[10];
rz(pi/2) q[10];
cx q[7],q[5];
cx q[5],q[7];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/4) q[5];
cx q[3],q[5];
rz(-pi/4) q[3];
rz(-pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[5];
rz(-pi/2) q[5];
rz(pi) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[11],q[9];
rz(-0.476842257587859) q[9];
cx q[11],q[9];
rz(-pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/4) q[9];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[13];
cx q[0],q[13];
cx q[13],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
sx q[13];
cx q[2],q[13];
rz(pi/4) q[13];
x q[2];
rz(pi/4) q[14];
rz(-2.2085455853959286) q[14];
rz(pi/2) q[14];
rz(pi/4) q[15];
rz(1.9091298816625877) q[16];
x q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/4) q[19];
rz(-pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[11],q[19];
rz(2.9570944897421336) q[19];
cx q[11],q[19];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(-0.4958017124672356) q[11];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
rz(1.7006627212057468) q[19];
sx q[19];
rz(8.571254204475807) q[19];
sx q[19];
rz(11.951934834800717) q[19];
rz(-pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[19],q[7];
rz(5.316913311237216) q[7];
cx q[19],q[7];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
cx q[20],q[17];
cx q[16],q[17];
rz(1.2790379364307158) q[17];
cx q[16],q[17];
rz(-pi/2) q[16];
rz(3.4353397104655894) q[16];
sx q[16];
rz(3.4342003958411187) q[16];
sx q[16];
rz(12.14142187969546) q[16];
rz(pi/2) q[17];
rz(-0.60609455521492) q[20];
sx q[20];
rz(8.220003921336055) q[20];
sx q[20];
rz(10.0308725159843) q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/4) q[20];
cx q[2],q[20];
rz(pi/4) q[2];
rz(-pi/4) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(-pi/2) q[20];
rz(-pi/2) q[20];
rz(-pi/2) q[20];
sx q[20];
rz(pi) q[20];
x q[20];
rz(3.5501314208624684) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[21];
cx q[12],q[21];
cx q[21],q[12];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(-pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[21],q[14];
rz(0) q[14];
sx q[14];
rz(0.6083733456447393) q[14];
sx q[14];
rz(3*pi) q[14];
rz(0) q[21];
sx q[21];
rz(5.6748119615348465) q[21];
sx q[21];
rz(3*pi) q[21];
cx q[21],q[14];
rz(-pi/2) q[14];
rz(2.2085455853959286) q[14];
rz(-pi/2) q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(4.466845776337103) q[21];
sx q[21];
rz(5*pi/2) q[21];
rz(pi/4) q[21];
cx q[21],q[8];
cx q[4],q[14];
rz(3.7767097314830123) q[14];
cx q[4],q[14];
rz(-pi/4) q[8];
cx q[21],q[8];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(1.1116701568981044) q[8];
rz(pi/2) q[22];
rz(2.895455706286576) q[22];
cx q[22],q[6];
rz(-2.895455706286576) q[6];
cx q[22],q[6];
rz(4.426929923075069) q[22];
cx q[22],q[11];
rz(-4.426929923075069) q[11];
sx q[11];
rz(2.3876543608423213) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[22],q[11];
rz(0) q[11];
sx q[11];
rz(3.895530946337265) q[11];
sx q[11];
rz(14.347509596311685) q[11];
sx q[11];
cx q[10],q[11];
x q[10];
rz(pi/2) q[11];
cx q[21],q[11];
cx q[11],q[21];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/2) q[11];
rz(4.542662735420071) q[21];
rz(3.855557285183768) q[21];
rz(pi/2) q[22];
rz(0.8100294969013189) q[22];
cx q[22],q[10];
rz(-0.8100294969013189) q[10];
cx q[22],q[10];
rz(0.8100294969013189) q[10];
rz(3.1132415027408094) q[10];
sx q[10];
rz(4.94265499985089) q[10];
rz(0) q[10];
sx q[10];
rz(5.209236225632083) q[10];
sx q[10];
rz(3*pi) q[10];
rz(-pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(2.895455706286576) q[6];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[15],q[23];
rz(-pi/4) q[23];
cx q[15],q[23];
cx q[15],q[18];
rz(5.522765531894115) q[18];
cx q[15],q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[15],q[18];
rz(3.8412140841069546) q[18];
cx q[15],q[18];
cx q[15],q[4];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/4) q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[1],q[23];
rz(5.6732579572932185) q[23];
cx q[1],q[23];
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
rz(-pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
x q[23];
cx q[23],q[14];
cx q[14],q[23];
sx q[23];
cx q[19],q[23];
x q[19];
rz(1.750377715766029) q[19];
sx q[19];
rz(pi/4) q[4];
rz(pi/2) q[24];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
rz(pi/2) q[24];
rz(2.6676373474942743) q[24];
sx q[24];
rz(6.792208934382879) q[24];
rz(1.1987968868385463) q[24];
rz(pi/2) q[24];
rz(pi/2) q[25];
rz(-0.39859910468378623) q[25];
sx q[25];
rz(6.71342751000741) q[25];
rz(pi/2) q[25];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[25],q[17];
cx q[17],q[25];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(-pi/4) q[17];
cx q[17],q[3];
rz(0.37118542855945025) q[25];
sx q[25];
rz(7.2288775277333785) q[25];
sx q[25];
rz(9.05359253220993) q[25];
rz(-0.5115571281857907) q[25];
sx q[25];
rz(5.401054688754893) q[25];
sx q[25];
rz(9.93633508895517) q[25];
cx q[3],q[17];
cx q[17],q[3];
rz(5.702486806977559) q[17];
rz(1.1418122462116176) q[3];
cx q[3],q[10];
rz(0) q[10];
sx q[10];
rz(1.0739490815475028) q[10];
sx q[10];
rz(3*pi) q[10];
cx q[3],q[10];
rz(pi) q[10];
rz(4.921450538321158) q[10];
sx q[10];
rz(5.526655321480697) q[10];
sx q[10];
rz(10.966136523674379) q[10];
rz(0) q[3];
sx q[3];
rz(3.2829016740994383) q[3];
sx q[3];
rz(3*pi) q[3];
rz(2.105963782184032) q[26];
cx q[26],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/4) q[26];
cx q[26],q[0];
rz(-pi/4) q[0];
cx q[26],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[24];
rz(0) q[0];
sx q[0];
rz(2.123849051923099) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[24];
sx q[24];
rz(2.123849051923099) q[24];
sx q[24];
rz(3*pi) q[24];
cx q[0],q[24];
rz(-pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[2],q[0];
rz(-pi/4) q[0];
cx q[2],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
cx q[13],q[0];
cx q[0],q[13];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[13];
sx q[13];
rz(8.660890178926746) q[13];
sx q[13];
rz(5*pi/2) q[13];
rz(1.480767377330423) q[13];
rz(pi/4) q[2];
rz(2.575935563182736) q[2];
cx q[2],q[23];
rz(-2.575935563182736) q[23];
cx q[2],q[23];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/4) q[2];
rz(2.575935563182736) q[23];
cx q[23],q[2];
rz(-pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-pi/2) q[2];
rz(-1.5387047371410594) q[2];
rz(pi/2) q[23];
rz(-pi/2) q[24];
rz(-1.1987968868385463) q[24];
cx q[24],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[24];
cx q[24],q[1];
rz(-pi/4) q[1];
cx q[24],q[1];
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
rz(-4.023589008567552) q[24];
sx q[24];
rz(6.374714178641874) q[24];
sx q[24];
rz(13.44836696933693) q[24];
rz(-1.3909777791919709) q[24];
rz(pi/2) q[24];
cx q[22],q[24];
rz(0) q[22];
sx q[22];
rz(4.452408341203062) q[22];
sx q[22];
rz(3*pi) q[22];
rz(0) q[24];
sx q[24];
rz(1.8307769659765234) q[24];
sx q[24];
rz(3*pi) q[24];
cx q[22],q[24];
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
rz(pi/2) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(-pi/2) q[24];
rz(1.3909777791919709) q[24];
rz(-3.665736859252118) q[24];
rz(pi/2) q[24];
cx q[7],q[1];
rz(-pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[5],q[1];
rz(6.194080711173252) q[1];
cx q[5],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(8.164170894097115) q[1];
sx q[1];
rz(5*pi/2) q[1];
sx q[1];
cx q[23],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
x q[23];
rz(-pi/2) q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[5],q[11];
cx q[11],q[5];
rz(pi/2) q[11];
rz(0.09613892765819357) q[5];
rz(1.0351851384223938) q[5];
rz(-pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(3.0555860712986584) q[7];
sx q[7];
rz(7.901395294654229) q[7];
sx q[7];
rz(14.173385844436341) q[7];
rz(0) q[7];
sx q[7];
rz(3.863425116745436) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[9],q[26];
rz(3.9970714754887045) q[26];
cx q[9],q[26];
rz(1.221805894562396) q[26];
rz(pi/2) q[26];
rz(-pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[26];
rz(0) q[26];
sx q[26];
rz(1.5620501427310425) q[26];
sx q[26];
rz(3*pi) q[26];
rz(0) q[9];
sx q[9];
rz(1.5620501427310425) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[9],q[26];
rz(-pi/2) q[26];
rz(-1.221805894562396) q[26];
rz(pi/2) q[26];
sx q[26];
rz(pi/2) q[26];
rz(-pi/2) q[26];
cx q[26],q[17];
rz(-pi/2) q[17];
rz(-pi/2) q[26];
cx q[13],q[26];
rz(-pi/2) q[13];
rz(3.8507729678416327) q[13];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[26];
rz(-pi/2) q[26];
rz(pi/2) q[26];
sx q[26];
rz(pi/2) q[26];
cx q[26],q[23];
rz(1.2993343791826726) q[23];
cx q[26],q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(pi/2) q[23];
rz(-pi/2) q[23];
rz(pi/2) q[26];
sx q[26];
rz(pi/2) q[26];
rz(pi/2) q[26];
rz(1.9888237166902967) q[26];
rz(-pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(0.9063344422334467) q[9];
cx q[16],q[9];
rz(-0.9063344422334467) q[9];
cx q[16],q[9];
cx q[16],q[25];
rz(2.309745041832841) q[25];
cx q[16],q[25];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-pi/2) q[25];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
rz(pi) q[9];
x q[9];
id q[9];
rz(-0.658688045225235) q[9];
sx q[9];
rz(7.197504492970936) q[9];
rz(2.5594725520383497) q[9];
rz(-pi/4) q[27];
rz(pi/2) q[27];
sx q[27];
rz(pi/2) q[27];
cx q[12],q[27];
rz(5.472321016776752) q[27];
cx q[12],q[27];
cx q[12],q[6];
rz(pi/2) q[27];
sx q[27];
rz(pi/2) q[27];
rz(pi/2) q[27];
rz(pi/2) q[27];
sx q[27];
rz(pi/2) q[27];
rz(pi/2) q[27];
rz(3.712509766406277) q[27];
sx q[27];
rz(6.681078745689323) q[27];
sx q[27];
rz(10.68863388711512) q[27];
rz(pi) q[27];
cx q[0],q[27];
cx q[27],q[0];
rz(6.196754877133291) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[19],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/4) q[19];
cx q[19],q[0];
rz(-pi/4) q[0];
cx q[19],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/2) q[19];
cx q[27],q[16];
rz(1.818581797052603) q[16];
cx q[27],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(2.655243146283018) q[16];
rz(4.378719509107524) q[16];
cx q[16],q[2];
rz(-4.378719509107524) q[2];
sx q[2];
rz(2.9118514346646505) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[16],q[2];
rz(-0.376318138161869) q[16];
sx q[16];
rz(3.6244105117820484) q[16];
rz(pi/4) q[16];
rz(0) q[2];
sx q[2];
rz(3.3713338725149358) q[2];
sx q[2];
rz(15.342202207017962) q[2];
rz(-pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[16],q[2];
rz(-pi/4) q[2];
cx q[16],q[2];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-pi/4) q[27];
cx q[6],q[12];
cx q[12],q[6];
rz(0.7561496933121057) q[12];
sx q[12];
rz(5.9876287632307506) q[12];
sx q[12];
rz(12.684943878801189) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[15],q[12];
rz(6.101298809884243) q[12];
cx q[15],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(2.897920370450284) q[12];
rz(-pi/4) q[12];
rz(pi) q[12];
x q[12];
rz(pi/2) q[12];
sx q[12];
rz(8.95222609011898) q[12];
sx q[12];
rz(5*pi/2) q[12];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[7];
rz(2.766694414585681) q[6];
rz(pi/2) q[6];
cx q[18],q[6];
rz(0) q[18];
sx q[18];
rz(1.4649178279478938) q[18];
sx q[18];
rz(3*pi) q[18];
rz(0) q[6];
sx q[6];
rz(1.4649178279478938) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[18],q[6];
rz(-pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
cx q[14],q[18];
cx q[14],q[4];
cx q[4],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[14],q[4];
rz(2.788181610099151) q[4];
cx q[14],q[4];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/4) q[14];
cx q[14],q[25];
rz(-pi/4) q[25];
cx q[14],q[25];
x q[14];
rz(pi/4) q[25];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(3.781762686458284) q[4];
sx q[4];
rz(6.3571032085194705) q[4];
sx q[4];
rz(14.687259633878881) q[4];
x q[4];
cx q[4],q[14];
cx q[14],q[4];
rz(-pi/2) q[6];
rz(-2.766694414585681) q[6];
rz(2.3045465026046457) q[6];
rz(0.35502194052429603) q[6];
cx q[21],q[6];
rz(-3.855557285183768) q[6];
sx q[6];
rz(2.265140259501287) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[21],q[6];
rz(-pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[21],q[24];
rz(0) q[21];
sx q[21];
rz(3.889687654019987) q[21];
sx q[21];
rz(3*pi) q[21];
rz(0) q[24];
sx q[24];
rz(2.3934976531595993) q[24];
sx q[24];
rz(3*pi) q[24];
cx q[21],q[24];
rz(-pi/2) q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
cx q[11],q[21];
x q[11];
cx q[21],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
sx q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(-pi/2) q[24];
rz(3.665736859252118) q[24];
cx q[24],q[3];
rz(0) q[3];
sx q[3];
rz(3.000283633080148) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[24],q[3];
rz(pi) q[24];
x q[24];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
cx q[3],q[19];
rz(pi/2) q[19];
rz(0.15468663490158707) q[19];
rz(pi/2) q[19];
rz(2.676390797872718) q[3];
cx q[5],q[19];
rz(0) q[19];
sx q[19];
rz(1.9849042059247448) q[19];
sx q[19];
rz(3*pi) q[19];
rz(0) q[5];
sx q[5];
rz(1.9849042059247448) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[5],q[19];
rz(-pi/2) q[19];
rz(-0.15468663490158707) q[19];
rz(1.9323476675073874) q[19];
rz(2.5765538082409245) q[19];
rz(0.17460754872717965) q[19];
sx q[19];
rz(1.8357715208014695) q[19];
rz(3.4757786824543007) q[19];
rz(-pi/2) q[19];
sx q[19];
rz(-pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(0) q[6];
sx q[6];
rz(4.018045047678299) q[6];
sx q[6];
rz(12.925313305428851) q[6];
rz(pi/2) q[6];
rz(0) q[7];
sx q[7];
rz(2.4197601904341504) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[15],q[7];
rz(1.2985872815255144) q[15];
sx q[15];
rz(3.181442212808463) q[15];
sx q[15];
rz(8.126190679243864) q[15];
rz(0.6166258265278357) q[15];
sx q[15];
rz(4.112865993145834) q[15];
sx q[15];
rz(13.43221725279086) q[15];
cx q[15],q[26];
rz(-1.9888237166902967) q[26];
cx q[15],q[26];
rz(-pi/2) q[26];
rz(pi/2) q[26];
rz(pi/2) q[26];
rz(0.1430692556148085) q[26];
rz(pi/4) q[7];
cx q[7],q[22];
rz(-pi/4) q[22];
cx q[7],q[22];
rz(pi/4) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[11],q[22];
rz(4.122316493577962) q[22];
cx q[11],q[22];
cx q[11],q[23];
rz(pi) q[11];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[23];
cx q[23],q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[20],q[7];
rz(2.457454847366744) q[7];
cx q[20],q[7];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(1.030034219556997) q[20];
cx q[20],q[22];
rz(-1.030034219556997) q[22];
cx q[20],q[22];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
rz(-pi/4) q[20];
rz(pi/4) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(1.030034219556997) q[22];
rz(5.254477507277212) q[22];
rz(-pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/4) q[7];
cx q[8],q[18];
rz(-1.1116701568981044) q[18];
cx q[8],q[18];
rz(1.1116701568981044) q[18];
sx q[18];
rz(pi/2) q[18];
sx q[8];
cx q[6],q[8];
x q[6];
cx q[6],q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[25],q[17];
rz(0.4548833960783466) q[17];
cx q[25],q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
rz(pi/2) q[25];
sx q[25];
rz(9.004101952852494) q[25];
sx q[25];
rz(5*pi/2) q[25];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[14],q[25];
rz(1.0439026149758708) q[25];
cx q[14],q[25];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
rz(pi/2) q[25];
rz(pi/2) q[25];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
rz(pi/4) q[6];
cx q[6],q[0];
rz(-pi/4) q[0];
cx q[6],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[24];
cx q[1],q[6];
rz(4.633434585426456) q[24];
cx q[0],q[24];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[5];
rz(5.754417325533375) q[0];
rz(pi/2) q[0];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
rz(pi) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(1.4003727073281158) q[5];
cx q[6],q[1];
cx q[1],q[16];
rz(2.9115593014202323) q[16];
cx q[1],q[16];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(1.0680783339312592) q[1];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-pi/2) q[16];
rz(4.3127941085693715) q[6];
sx q[6];
rz(4.096323932480018) q[6];
rz(0.32755853397767415) q[6];
cx q[11],q[6];
rz(-0.32755853397767415) q[6];
cx q[11],q[6];
rz(-pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[7],q[17];
rz(-pi/4) q[17];
cx q[7],q[17];
rz(pi/4) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[15],q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(-pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[17],q[0];
rz(0) q[0];
sx q[0];
rz(0.9633082674988991) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[17];
sx q[17];
rz(0.9633082674988991) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[17],q[0];
rz(-pi/2) q[0];
rz(-5.754417325533375) q[0];
rz(-pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(-pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[11],q[17];
rz(3.2181971559821627) q[17];
cx q[11],q[17];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(-0.2269052787406213) q[7];
rz(pi/2) q[7];
cx q[13],q[7];
rz(0) q[13];
sx q[13];
rz(3.756996985604753) q[13];
sx q[13];
rz(3*pi) q[13];
rz(0) q[7];
sx q[7];
rz(2.5261883215748333) q[7];
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
rz(-pi/2) q[7];
rz(0.2269052787406213) q[7];
cx q[7],q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/4) q[16];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(0) q[7];
sx q[7];
rz(4.197907906735496) q[7];
sx q[7];
rz(3*pi) q[7];
rz(2.2950826908568436) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[27],q[8];
rz(5.514707948886041) q[8];
cx q[27],q[8];
rz(0.9448575337501297) q[27];
sx q[27];
rz(2.273036585561056) q[27];
cx q[24],q[27];
cx q[27],q[24];
rz(0) q[24];
sx q[24];
rz(4.397571936796264) q[24];
sx q[24];
rz(3*pi) q[24];
cx q[14],q[24];
rz(0) q[24];
sx q[24];
rz(1.8856133703833224) q[24];
sx q[24];
rz(3*pi) q[24];
cx q[14],q[24];
cx q[14],q[16];
rz(-pi/4) q[14];
rz(pi/4) q[14];
rz(-pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-pi/2) q[16];
rz(-pi/4) q[16];
sx q[16];
rz(5.470383484027694) q[16];
rz(1.3706506766109499) q[24];
sx q[24];
rz(4.103505358538875) q[24];
rz(0) q[24];
sx q[24];
rz(3.517968077122665) q[24];
sx q[24];
rz(3*pi) q[24];
cx q[11],q[24];
rz(0) q[24];
sx q[24];
rz(2.7652172300569213) q[24];
sx q[24];
rz(3*pi) q[24];
cx q[11],q[24];
rz(0.215453410177659) q[27];
rz(-pi/2) q[27];
rz(pi/2) q[27];
sx q[27];
rz(pi/2) q[27];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[3];
rz(-2.676390797872718) q[3];
cx q[8],q[3];
rz(0.3044752643949433) q[8];
sx q[8];
rz(6.041599238255047) q[8];
sx q[8];
rz(9.120302696374436) q[8];
rz(-4.118504922715071) q[8];
rz(pi/2) q[8];
cx q[22],q[8];
rz(0) q[22];
sx q[22];
rz(6.155786647397422) q[22];
sx q[22];
rz(3*pi) q[22];
rz(0) q[8];
sx q[8];
rz(0.1273986597821648) q[8];
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
cx q[0],q[22];
rz(-0.20979968772764312) q[0];
sx q[0];
rz(4.200477238015109) q[0];
sx q[0];
rz(9.634577648497022) q[0];
rz(-pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/4) q[22];
rz(pi/4) q[22];
rz(-pi/2) q[8];
rz(4.118504922715071) q[8];
cx q[8],q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/4) q[8];
cx q[8],q[6];
rz(-pi/4) q[6];
cx q[8],q[6];
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
rz(-pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[0],q[6];
rz(3.7018143813425914) q[6];
cx q[0],q[6];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(0.4249992477207247) q[0];
sx q[0];
rz(4.39159728470364) q[0];
sx q[0];
rz(15.565843336140091) q[0];
rz(-0.01874480104586551) q[0];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(3.244944252811715) q[8];
sx q[8];
rz(2.2314082209623396) q[8];
rz(pi/4) q[8];
rz(-pi/2) q[8];
cx q[9],q[18];
rz(-2.5594725520383497) q[18];
cx q[9],q[18];
rz(2.5594725520383497) q[18];
cx q[18],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[2],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
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
rz(pi) q[2];
x q[2];
rz(pi/2) q[2];
sx q[2];
rz(8.521501555365738) q[2];
sx q[2];
rz(5*pi/2) q[2];
cx q[21],q[12];
rz(1.2280909765744326) q[12];
cx q[21],q[12];
rz(pi) q[12];
cx q[17],q[12];
cx q[12],q[17];
rz(-pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[21],q[27];
rz(2.5695406524093927) q[27];
cx q[21],q[27];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
rz(pi/2) q[27];
sx q[27];
rz(pi/2) q[27];
rz(pi/2) q[27];
cx q[27],q[7];
cx q[4],q[18];
cx q[13],q[4];
rz(2.5142201286429353) q[13];
sx q[13];
rz(4.6901098224426825) q[13];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
id q[18];
rz(pi/2) q[18];
cx q[21],q[13];
rz(5.484583570564744) q[13];
cx q[21],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(2.843733059923199) q[13];
cx q[17],q[13];
rz(-2.843733059923199) q[13];
cx q[17],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(0.31482117269630816) q[17];
sx q[17];
rz(3.9669140904614433) q[17];
sx q[17];
rz(9.10995678807307) q[17];
rz(pi/2) q[21];
cx q[25],q[18];
cx q[18],q[25];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(2.9115099698233378) q[25];
sx q[25];
rz(7.236326963290462) q[25];
sx q[25];
rz(13.928254176079818) q[25];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[14],q[25];
rz(-pi/4) q[25];
cx q[14],q[25];
rz(pi/2) q[14];
cx q[14],q[19];
x q[14];
rz(-4.568896358826811) q[14];
sx q[14];
rz(4.167904074626213) q[14];
sx q[14];
rz(13.993674319596192) q[14];
rz(-pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-2.625074697376837) q[19];
sx q[19];
rz(3.421506156658092) q[19];
sx q[19];
rz(12.049852658146216) q[19];
rz(pi/4) q[19];
rz(-pi/4) q[19];
rz(2.976601047954413) q[19];
sx q[19];
rz(4.460086221368754) q[19];
sx q[19];
rz(9.83503266960884) q[19];
rz(pi/4) q[25];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
rz(pi/4) q[25];
rz(1.0002237965167333) q[4];
cx q[1],q[4];
rz(-1.0680783339312592) q[4];
sx q[4];
rz(1.4700739348243759) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[1],q[4];
rz(0) q[4];
sx q[4];
rz(4.813111372355211) q[4];
sx q[4];
rz(9.492632498183905) q[4];
cx q[26],q[4];
rz(-0.1430692556148085) q[4];
cx q[26],q[4];
rz(pi/2) q[26];
rz(pi/2) q[26];
sx q[26];
rz(pi/2) q[26];
rz(0.1430692556148085) q[4];
cx q[12],q[4];
rz(1.5120160027273761) q[12];
rz(1.692171708439083) q[4];
cx q[13],q[4];
rz(-1.692171708439083) q[4];
cx q[13],q[4];
cx q[17],q[13];
rz(2.502951010032474) q[13];
cx q[17],q[13];
rz(-pi/4) q[13];
rz(0) q[7];
sx q[7];
rz(2.0852774004440904) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[27],q[7];
rz(-1.3328176588995153) q[27];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[24],q[7];
rz(3.3362902811742323) q[7];
cx q[24],q[7];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
cx q[25],q[24];
rz(-pi/4) q[24];
cx q[25],q[24];
rz(pi/4) q[24];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[8];
rz(pi/2) q[8];
rz(0) q[8];
sx q[8];
rz(9.195668863483563) q[8];
sx q[8];
rz(3*pi) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[10],q[9];
rz(1.168738745137768) q[9];
cx q[10],q[9];
cx q[10],q[3];
cx q[3],q[10];
cx q[10],q[3];
rz(2.9661958638288066) q[10];
cx q[10],q[15];
rz(-2.9661958638288066) q[15];
cx q[10],q[15];
rz(2.9661958638288066) q[15];
rz(1.4518447390614608) q[15];
cx q[15],q[10];
rz(-1.4518447390614608) q[10];
cx q[15],q[10];
rz(1.4518447390614608) q[10];
cx q[10],q[2];
cx q[15],q[20];
rz(4.678624578364717) q[15];
cx q[15],q[27];
rz(4.444367474780107) q[2];
cx q[10],q[2];
rz(-pi/2) q[10];
cx q[11],q[10];
rz(pi/2) q[10];
rz(-2.2678599890779476) q[10];
rz(0) q[11];
sx q[11];
rz(6.71591788381558) q[11];
sx q[11];
rz(3*pi) q[11];
sx q[11];
rz(pi) q[11];
x q[11];
rz(3.5521948695908563) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[20],q[5];
rz(-4.678624578364717) q[27];
sx q[27];
rz(3.086514743612731) q[27];
sx q[27];
rz(3*pi) q[27];
cx q[15],q[27];
rz(5.123157071197366) q[15];
rz(3.361293881172171) q[15];
cx q[15],q[10];
rz(-3.361293881172171) q[10];
sx q[10];
rz(0.7536505812985328) q[10];
sx q[10];
rz(3*pi) q[10];
cx q[15],q[10];
rz(0) q[10];
sx q[10];
rz(5.529534725881053) q[10];
sx q[10];
rz(15.053931831019497) q[10];
rz(1.131599960455972) q[10];
sx q[10];
rz(6.1062210262231575) q[10];
rz(pi/4) q[15];
rz(0) q[27];
sx q[27];
rz(3.196670563566855) q[27];
sx q[27];
rz(15.436220198033613) q[27];
rz(pi/2) q[27];
sx q[27];
rz(8.821216978397288) q[27];
sx q[27];
rz(5*pi/2) q[27];
rz(2.1886618095745374) q[27];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[23],q[3];
rz(pi/4) q[23];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[23],q[3];
rz(-pi/4) q[3];
cx q[23],q[3];
rz(-2.1131366983006807) q[23];
sx q[23];
rz(4.872193016059925) q[23];
sx q[23];
rz(11.53791465907006) q[23];
rz(5.712053850074374) q[23];
cx q[23],q[18];
rz(4.4773868131976835) q[18];
cx q[23],q[18];
rz(-pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(0.4763234215673733) q[3];
rz(0) q[3];
sx q[3];
rz(3.535195445312681) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[1],q[3];
rz(0) q[3];
sx q[3];
rz(2.7479898618669054) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[1],q[3];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[22],q[1];
rz(-pi/4) q[1];
cx q[22],q[1];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[21],q[22];
rz(pi/4) q[21];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[21],q[22];
rz(-pi/4) q[22];
cx q[21],q[22];
rz(pi/4) q[21];
rz(pi/4) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[23],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
id q[1];
rz(-pi/4) q[1];
cx q[23],q[24];
rz(5.206098758123799) q[24];
cx q[23],q[24];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
cx q[21],q[24];
cx q[24],q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[2],q[3];
rz(5.742878701782634) q[3];
cx q[2],q[3];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
cx q[3],q[12];
rz(-1.5120160027273761) q[12];
cx q[3],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[15],q[12];
rz(-pi/4) q[12];
cx q[15],q[12];
rz(pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi) q[15];
x q[15];
cx q[15],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[15],q[11];
rz(0.6028860569436812) q[11];
cx q[15],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi) q[3];
rz(-1.4003727073281158) q[5];
cx q[20],q[5];
rz(-pi/4) q[20];
rz(6.056012005820583) q[20];
rz(-pi/2) q[20];
cx q[17],q[20];
rz(pi/2) q[17];
x q[17];
rz(pi/2) q[20];
id q[20];
rz(-2.7010598293352106) q[20];
rz(pi/2) q[20];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
sx q[5];
rz(1.6378977883332302) q[5];
rz(3.2749219450371005) q[5];
cx q[5],q[0];
rz(-3.2749219450371005) q[0];
sx q[0];
rz(2.0899457512380595) q[0];
sx q[0];
rz(3*pi) q[0];
cx q[5],q[0];
rz(0) q[0];
sx q[0];
rz(4.193239555941527) q[0];
sx q[0];
rz(12.718444706852345) q[0];
rz(-pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[14];
rz(2.3021808243252626) q[14];
cx q[0],q[14];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(-pi/4) q[0];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(3.565505759010642) q[14];
sx q[14];
rz(5*pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[0],q[14];
rz(5.311831659696135) q[14];
cx q[0],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/4) q[5];
rz(-pi/2) q[5];
rz(1.3650218584703155) q[5];
rz(0.46853989094998283) q[5];
cx q[6],q[2];
cx q[2],q[6];
cx q[6],q[2];
cx q[4],q[2];
rz(6.037545476374176) q[2];
cx q[4],q[2];
rz(3.5257610555851877) q[4];
cx q[4],q[24];
cx q[24],q[4];
cx q[4],q[24];
rz(0.1695405271161823) q[4];
cx q[4],q[24];
rz(-0.1695405271161823) q[24];
cx q[4],q[24];
rz(0.1695405271161823) q[24];
rz(pi/4) q[6];
cx q[6],q[22];
rz(-pi/4) q[22];
cx q[6],q[22];
rz(pi/4) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[2],q[22];
rz(3.751972825788834) q[22];
cx q[2],q[22];
cx q[2],q[1];
rz(5.611954152511292) q[1];
cx q[2],q[1];
rz(-0.16339105091215567) q[1];
rz(0) q[2];
sx q[2];
rz(5.7408743032071765) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[5],q[1];
rz(-0.46853989094998283) q[1];
sx q[1];
rz(0.6230208853043848) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[5],q[1];
rz(0) q[1];
sx q[1];
rz(5.660164421875201) q[1];
sx q[1];
rz(10.056708902631518) q[1];
rz(0) q[6];
sx q[6];
rz(4.360650486393295) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[7],q[27];
rz(-2.1886618095745374) q[27];
cx q[7],q[27];
cx q[23],q[7];
rz(-4.8883845129483445) q[27];
rz(pi/2) q[27];
cx q[12],q[27];
rz(0) q[12];
sx q[12];
rz(4.146268068252861) q[12];
sx q[12];
rz(3*pi) q[12];
rz(0) q[27];
sx q[27];
rz(2.136917238926725) q[27];
sx q[27];
rz(3*pi) q[27];
cx q[12],q[27];
rz(-pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(2.855709355797414) q[12];
cx q[12],q[22];
rz(-2.855709355797414) q[22];
cx q[12],q[22];
rz(1.2274411910325786) q[12];
sx q[12];
rz(9.102179817169926) q[12];
sx q[12];
rz(11.72468732503957) q[12];
rz(2.855709355797414) q[22];
rz(-pi/4) q[22];
rz(-pi/2) q[27];
rz(4.8883845129483445) q[27];
rz(3.0862732614182335) q[27];
sx q[27];
rz(7.222930605874274) q[27];
sx q[27];
rz(10.021400362884723) q[27];
id q[27];
rz(0.06040900533684242) q[7];
cx q[23],q[7];
rz(3.1302445277059094) q[23];
rz(1.2211595859752873) q[7];
sx q[7];
rz(8.888133273857028) q[7];
sx q[7];
rz(13.589282115719655) q[7];
rz(pi/2) q[7];
cx q[8],q[10];
cx q[10],q[8];
cx q[8],q[10];
rz(pi/4) q[10];
cx q[10],q[21];
rz(-pi/4) q[21];
cx q[10],q[21];
sx q[10];
rz(pi/4) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(-pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[21],q[20];
rz(0) q[20];
sx q[20];
rz(1.961230420278054) q[20];
sx q[20];
rz(3*pi) q[20];
rz(0) q[21];
sx q[21];
rz(4.3219548869015325) q[21];
sx q[21];
rz(3*pi) q[21];
cx q[21],q[20];
rz(-pi/2) q[20];
rz(2.7010598293352106) q[20];
rz(-pi/2) q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
rz(pi/4) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(1.1359045764781115) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(-pi/4) q[9];
rz(pi/2) q[9];
cx q[26],q[9];
cx q[9],q[26];
rz(-pi/2) q[26];
rz(pi/2) q[26];
sx q[26];
rz(pi/2) q[26];
cx q[18],q[26];
rz(5.792235277489477) q[26];
cx q[18],q[26];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/4) q[18];
cx q[16],q[18];
rz(-0.7019180085891317) q[16];
rz(-pi/4) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi/2) q[18];
rz(4.606850787105539) q[18];
rz(3.8947826496636044) q[18];
cx q[18],q[16];
rz(-3.8947826496636044) q[16];
sx q[16];
rz(1.1801850017713127) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[18],q[16];
rz(0) q[16];
sx q[16];
rz(5.1030003054082735) q[16];
sx q[16];
rz(14.021478619022115) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(1.9790816734815366) q[18];
rz(pi/2) q[26];
sx q[26];
rz(pi/2) q[26];
rz(pi/2) q[26];
rz(2.364692125056815) q[26];
sx q[26];
rz(5.564583457509045) q[26];
cx q[26],q[6];
rz(0) q[6];
sx q[6];
rz(1.922534820786291) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[26],q[6];
rz(0) q[26];
sx q[26];
rz(4.983536214742493) q[26];
sx q[26];
rz(3*pi) q[26];
rz(3.8197599627971943) q[26];
cx q[6],q[23];
rz(-3.1302445277059094) q[23];
cx q[6],q[23];
rz(-pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(-3.36257500983131) q[6];
rz(pi/2) q[6];
cx q[23],q[6];
rz(0) q[23];
sx q[23];
rz(4.782378757853051) q[23];
sx q[23];
rz(3*pi) q[23];
rz(0) q[6];
sx q[6];
rz(1.5008065493265357) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[23],q[6];
rz(-pi/2) q[23];
rz(pi/2) q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(pi/2) q[23];
rz(pi/2) q[23];
rz(-pi/2) q[6];
rz(3.36257500983131) q[6];
cx q[8],q[16];
rz(-pi/4) q[16];
cx q[8],q[16];
rz(pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[13];
rz(3.407753854190415) q[13];
cx q[16],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
sx q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(0) q[9];
sx q[9];
rz(7.936791531839916) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[25],q[9];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[3],q[25];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[3],q[25];
cx q[25],q[3];
cx q[3],q[25];
rz(3.532739337608052) q[25];
sx q[25];
rz(4.645189506923739) q[25];
sx q[25];
rz(12.784939900955465) q[25];
rz(-pi/4) q[9];
cx q[18],q[9];
rz(-1.9790816734815366) q[9];
cx q[18],q[9];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(2.562379992920998) q[18];
cx q[3],q[18];
rz(-2.562379992920998) q[18];
cx q[3],q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(1.9790816734815366) q[9];
id q[9];
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
measure q[24] -> c[24];
measure q[25] -> c[25];
measure q[26] -> c[26];
measure q[27] -> c[27];