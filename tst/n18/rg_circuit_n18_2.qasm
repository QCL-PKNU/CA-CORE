OPENQASM 2.0;
include "qelib1.inc";
qreg q[18];
creg c[18];
rz(-pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/2) q[1];
rz(-0.9664220514072863) q[2];
rz(pi/4) q[5];
id q[5];
rz(pi/4) q[5];
rz(1.2175613939039391) q[7];
rz(pi) q[7];
rz(-pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[0];
rz(2.795030596950206) q[0];
cx q[8],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(4.628481820910547) q[0];
sx q[0];
rz(5*pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(-pi/2) q[8];
rz(4.337934025423832) q[9];
cx q[9],q[2];
rz(-4.337934025423832) q[2];
sx q[2];
rz(1.7478359189523496) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[9],q[2];
rz(0) q[2];
sx q[2];
rz(4.535349388227237) q[2];
sx q[2];
rz(14.729134037600497) q[2];
cx q[10],q[1];
rz(pi/2) q[1];
rz(-0.4265320427296777) q[1];
sx q[1];
rz(8.75023975743405) q[1];
sx q[1];
rz(9.851310003499057) q[1];
cx q[2],q[10];
rz(-pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[11];
cx q[3],q[11];
rz(-pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/2) q[11];
rz(5.590809929359038) q[11];
rz(pi/2) q[11];
rz(-1.8556497624218502) q[3];
sx q[3];
rz(3.7543724560280367) q[3];
sx q[3];
rz(11.280427723191229) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/4) q[3];
cx q[8],q[3];
rz(-pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[3];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[6],q[12];
rz(5.624010056987891) q[12];
cx q[6],q[12];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[11];
rz(0) q[11];
sx q[11];
rz(0.2683902282360555) q[11];
sx q[11];
rz(3*pi) q[11];
rz(0) q[12];
sx q[12];
rz(0.2683902282360555) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[12],q[11];
rz(-pi/2) q[11];
rz(-5.590809929359038) q[11];
rz(pi/2) q[11];
rz(-pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
cx q[1],q[12];
cx q[12],q[1];
rz(pi/2) q[1];
rz(2.938127512714896) q[1];
rz(pi/2) q[12];
rz(-pi/2) q[12];
rz(-pi/4) q[6];
cx q[6],q[2];
cx q[2],q[6];
cx q[6],q[2];
rz(pi/4) q[6];
cx q[6],q[8];
rz(-pi/4) q[8];
cx q[6],q[8];
rz(-pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
id q[13];
cx q[13],q[9];
rz(0.6411937087875612) q[9];
cx q[13],q[9];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[5],q[13];
rz(-pi/4) q[13];
cx q[5],q[13];
rz(pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(0.15074676183073915) q[13];
cx q[13],q[11];
rz(-0.15074676183073915) q[11];
cx q[13],q[11];
rz(0.15074676183073915) q[11];
rz(0.0353547260504703) q[11];
rz(2.3942700081154484) q[13];
rz(pi/2) q[13];
cx q[2],q[5];
cx q[5],q[2];
cx q[2],q[5];
cx q[2],q[12];
rz(pi/2) q[12];
rz(-pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-2.9486217627400384) q[2];
rz(pi/2) q[2];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(2.053546301565167) q[5];
cx q[6],q[13];
rz(0) q[13];
sx q[13];
rz(1.9593672584921629) q[13];
sx q[13];
rz(3*pi) q[13];
rz(0) q[6];
sx q[6];
rz(1.9593672584921629) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[6],q[13];
rz(-pi/2) q[13];
rz(-2.3942700081154484) q[13];
rz(pi/4) q[13];
rz(-pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(5.833402687764516) q[6];
rz(pi/2) q[6];
rz(-pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[10];
rz(5.931439629929708) q[10];
cx q[9],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(-pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(-pi/4) q[9];
rz(1.3658946086840396) q[9];
rz(5.20569419693552) q[9];
cx q[9],q[11];
rz(-5.20569419693552) q[11];
sx q[11];
rz(2.654614331046912) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[9],q[11];
rz(0) q[11];
sx q[11];
rz(3.628570976132674) q[11];
sx q[11];
rz(14.59511743165443) q[11];
rz(5.782488279555496) q[11];
rz(pi/2) q[11];
rz(-pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(1.0548088546519678) q[9];
x q[14];
rz(pi/2) q[14];
id q[15];
rz(-1.255296762699341) q[15];
sx q[15];
rz(6.4826510226799074) q[15];
sx q[15];
rz(10.68007472346872) q[15];
rz(pi/2) q[15];
rz(-4.50857633025188) q[16];
sx q[16];
rz(4.484736559232149) q[16];
sx q[16];
rz(13.93335429102126) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[14];
cx q[14],q[16];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[15];
cx q[15],q[16];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(-pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[10];
rz(2.3638472375323367) q[10];
cx q[16],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
id q[10];
rz(-pi/2) q[10];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[3],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/4) q[3];
cx q[3],q[15];
rz(-pi/4) q[15];
cx q[3],q[15];
rz(pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[5];
rz(pi) q[3];
rz(-pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[2];
rz(0) q[2];
sx q[2];
rz(1.8283040393218903) q[2];
sx q[2];
rz(3*pi) q[2];
rz(0) q[3];
sx q[3];
rz(4.4548812678576954) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[3],q[2];
rz(-pi/2) q[2];
rz(2.9486217627400384) q[2];
rz(-pi/2) q[2];
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
rz(-2.053546301565167) q[5];
cx q[15],q[5];
rz(-pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[5],q[6];
rz(0) q[5];
sx q[5];
rz(0.674774278544092) q[5];
sx q[5];
rz(3*pi) q[5];
rz(0) q[6];
sx q[6];
rz(0.674774278544092) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[5],q[6];
rz(-pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(-pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[5],q[2];
rz(0.7251481108016702) q[2];
cx q[5],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(1.916897571508383) q[2];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[6];
rz(-5.833402687764516) q[6];
rz(0.2715356366580849) q[6];
cx q[6],q[10];
rz(-0.2715356366580849) q[10];
cx q[6],q[10];
rz(0.2715356366580849) q[10];
x q[6];
cx q[7],q[14];
rz(1.0350846097918562) q[14];
cx q[7],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(-pi/2) q[14];
rz(pi) q[14];
x q[14];
rz(pi) q[14];
rz(0.3092189299243946) q[14];
cx q[14],q[3];
rz(-0.3092189299243946) q[3];
cx q[14],q[3];
sx q[14];
rz(0.3092189299243946) q[3];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi) q[7];
x q[7];
rz(-1.2349662344589643) q[7];
cx q[1],q[7];
rz(-2.938127512714896) q[7];
sx q[7];
rz(1.444412790300021) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[1],q[7];
rz(-pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[1],q[11];
rz(0) q[1];
sx q[1];
rz(0.8125987498834055) q[1];
sx q[1];
rz(3*pi) q[1];
rz(0) q[11];
sx q[11];
rz(0.8125987498834055) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[1],q[11];
rz(-pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(-pi/2) q[11];
rz(-5.782488279555496) q[11];
rz(-pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(0) q[7];
sx q[7];
rz(4.838772516879565) q[7];
sx q[7];
rz(13.59787170794324) q[7];
rz(4.394226694343271) q[7];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[4],q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(4.579084245514994) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[4];
sx q[4];
rz(5.327622876499783) q[4];
sx q[4];
rz(5*pi/2) q[4];
rz(pi/2) q[4];
cx q[17],q[4];
cx q[4],q[17];
rz(pi/2) q[17];
sx q[17];
rz(4.427053040895464) q[17];
sx q[17];
rz(5*pi/2) q[17];
sx q[17];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
sx q[4];
cx q[0],q[4];
x q[0];
rz(pi/4) q[0];
cx q[0],q[16];
rz(-pi/4) q[16];
cx q[0],q[16];
rz(-pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[12];
rz(1.054528162704246) q[12];
cx q[0],q[12];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(2.324271588202958) q[0];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(1.5472520318653429) q[12];
rz(0.44807115743025044) q[12];
rz(pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
sx q[16];
cx q[16],q[1];
cx q[1],q[16];
cx q[16],q[1];
rz(pi/2) q[1];
cx q[1],q[14];
x q[1];
rz(1.8331133407053875) q[1];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/4) q[16];
cx q[0],q[16];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-pi/2) q[16];
id q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-4.351658169795873) q[4];
sx q[4];
rz(7.193789838381936) q[4];
sx q[4];
rz(13.776436130565251) q[4];
rz(4.830362257748438) q[4];
rz(2.6359149384053073) q[4];
cx q[4],q[2];
rz(-2.6359149384053073) q[2];
sx q[2];
rz(2.7961651424022573) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[4],q[2];
rz(0) q[2];
sx q[2];
rz(3.487020164777329) q[2];
sx q[2];
rz(10.143795327666304) q[2];
rz(0) q[2];
sx q[2];
rz(5.848962175677525) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[14],q[2];
rz(0) q[2];
sx q[2];
rz(0.43422313150206104) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[14],q[2];
rz(5.222968378763597) q[2];
sx q[2];
rz(5.217366565439891) q[2];
sx q[2];
rz(9.730691843119532) q[2];
rz(5.8020586120811455) q[2];
rz(pi/2) q[2];
rz(pi/4) q[4];
cx q[4],q[5];
rz(-pi/4) q[5];
cx q[4],q[5];
sx q[4];
rz(-pi/2) q[4];
rz(-pi/2) q[4];
rz(pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
id q[5];
cx q[8],q[17];
rz(-pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[15],q[17];
rz(4.277534147115066) q[17];
cx q[15],q[17];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(pi) q[15];
x q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(-pi/4) q[17];
rz(-0.34188315271712294) q[17];
cx q[12],q[17];
rz(-0.44807115743025044) q[17];
sx q[17];
rz(1.6315133194106852) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[12],q[17];
rz(0) q[17];
sx q[17];
rz(4.651671987768901) q[17];
sx q[17];
rz(10.214732270916752) q[17];
x q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[13],q[8];
rz(-pi/4) q[8];
cx q[13],q[8];
cx q[13],q[7];
cx q[7],q[13];
cx q[13],q[7];
cx q[13],q[9];
rz(pi/4) q[7];
cx q[7],q[15];
rz(-pi/4) q[15];
cx q[7],q[15];
cx q[1],q[7];
rz(pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[16],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(-pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
sx q[16];
rz(4.051992413037945) q[16];
sx q[16];
rz(2.8772859720364075) q[16];
sx q[16];
rz(pi/4) q[16];
id q[16];
rz(4.261934324555408) q[16];
rz(-1.8331133407053875) q[7];
cx q[1],q[7];
rz(-0.6232933373420622) q[1];
sx q[1];
rz(7.325055386034964) q[1];
rz(1.8331133407053875) q[7];
rz(4.137988952297741) q[7];
rz(pi/2) q[7];
cx q[15],q[7];
rz(0) q[15];
sx q[15];
rz(1.9158038142059641) q[15];
sx q[15];
rz(3*pi) q[15];
rz(0) q[7];
sx q[7];
rz(1.9158038142059641) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[15],q[7];
rz(-pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(-pi/2) q[7];
rz(-4.137988952297741) q[7];
rz(2.968185252651634) q[7];
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[11],q[8];
rz(5.492860748145065) q[8];
cx q[11],q[8];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
cx q[11],q[10];
rz(6.04713104055036) q[10];
cx q[11],q[10];
rz(pi/4) q[10];
cx q[10],q[0];
rz(-pi/4) q[0];
cx q[10],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi) q[0];
x q[0];
rz(-pi/2) q[0];
rz(pi/4) q[10];
rz(-pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[2];
rz(0) q[10];
sx q[10];
rz(1.1429729733428127) q[10];
sx q[10];
rz(3*pi) q[10];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(1.898098439627793) q[11];
cx q[12],q[11];
rz(-1.898098439627793) q[11];
cx q[12],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(1.1984462494123467) q[11];
sx q[12];
cx q[15],q[0];
rz(pi/2) q[0];
rz(0) q[0];
sx q[0];
rz(9.360212425481247) q[0];
sx q[0];
rz(3*pi) q[0];
rz(pi/4) q[0];
rz(4.375559628976329) q[15];
sx q[15];
rz(4.204691869809546) q[15];
sx q[15];
rz(14.686203207795247) q[15];
rz(0.513026225291281) q[15];
cx q[15],q[4];
rz(0) q[2];
sx q[2];
rz(1.1429729733428127) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[10],q[2];
rz(-pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
id q[10];
rz(pi/4) q[10];
rz(4.91068614569469) q[10];
sx q[10];
rz(6.070174695348746) q[10];
sx q[10];
rz(14.310936498614543) q[10];
id q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-pi/2) q[2];
rz(-5.8020586120811455) q[2];
rz(2.2010159932694986) q[2];
rz(-0.513026225291281) q[4];
cx q[15],q[4];
rz(-pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(0.513026225291281) q[4];
rz(3.3219396002265498) q[4];
rz(0) q[4];
sx q[4];
rz(4.056587527915601) q[4];
sx q[4];
rz(3*pi) q[4];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
cx q[8],q[3];
cx q[3],q[8];
cx q[8],q[3];
cx q[3],q[6];
rz(3.1543872417392818) q[6];
cx q[3],q[6];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[14],q[3];
rz(pi/4) q[14];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[14],q[3];
rz(-pi/4) q[3];
cx q[14],q[3];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/4) q[14];
rz(pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[14];
rz(-pi/4) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/2) q[14];
rz(0.8556985182586128) q[14];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-4.351253695813555) q[6];
sx q[6];
rz(7.332619661144791) q[6];
sx q[6];
rz(13.776031656582933) q[6];
cx q[7],q[6];
rz(-2.968185252651634) q[6];
cx q[7],q[6];
rz(2.968185252651634) q[6];
x q[6];
rz(-pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(-pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-1.0548088546519678) q[9];
cx q[13],q[9];
cx q[13],q[17];
rz(2.0476170879371227) q[13];
sx q[13];
rz(4.91603899286784) q[13];
sx q[13];
rz(9.769815334805532) q[13];
rz(-0.6146820279956168) q[13];
sx q[13];
rz(7.503584220139773) q[13];
id q[13];
rz(-pi/2) q[13];
rz(4.413599203162834) q[17];
rz(4.69937767615647) q[17];
cx q[17],q[11];
rz(-4.69937767615647) q[11];
sx q[11];
rz(1.1289340027670454) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[17],q[11];
rz(0) q[11];
sx q[11];
rz(5.154251304412541) q[11];
sx q[11];
rz(12.925709387513502) q[11];
rz(-pi/2) q[11];
rz(2.049990087320758) q[11];
sx q[11];
rz(2.9497543495490346) q[11];
rz(-0.4128386151580652) q[11];
rz(pi/2) q[11];
cx q[17],q[5];
cx q[3],q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[5],q[17];
cx q[17],q[5];
rz(pi) q[17];
rz(0) q[17];
sx q[17];
rz(3.40373422118449) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[2],q[17];
rz(0) q[17];
sx q[17];
rz(2.8794510859950964) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[2],q[17];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(1.2781326897933514) q[2];
rz(-pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[6],q[11];
rz(0) q[11];
sx q[11];
rz(0.6551389641039975) q[11];
sx q[11];
rz(3*pi) q[11];
rz(0) q[6];
sx q[6];
rz(5.628046343075589) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[6],q[11];
rz(-pi/2) q[11];
rz(0.4128386151580652) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[17],q[11];
rz(4.886369684086991) q[11];
cx q[17],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[11];
rz(-pi/2) q[11];
rz(pi) q[17];
x q[17];
rz(-pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
cx q[7],q[5];
rz(3.0125925328113015) q[5];
cx q[7],q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
cx q[7],q[5];
rz(4.4359598563175275) q[5];
rz(1.1830564220796815) q[5];
rz(2.726520129644137) q[7];
rz(pi/2) q[7];
cx q[15],q[7];
rz(0) q[15];
sx q[15];
rz(0.9976621403533179) q[15];
sx q[15];
rz(3*pi) q[15];
rz(0) q[7];
sx q[7];
rz(0.9976621403533179) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[15],q[7];
rz(-pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(pi) q[15];
rz(-pi/2) q[7];
rz(-2.726520129644137) q[7];
rz(1.9491180881408139) q[7];
cx q[7],q[2];
rz(-1.9491180881408139) q[2];
sx q[2];
rz(0.10552964070410642) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[7],q[2];
rz(0) q[2];
sx q[2];
rz(6.17765566647548) q[2];
sx q[2];
rz(10.095763359116843) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[7],q[17];
cx q[17],q[7];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi) q[7];
x q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(-6.069096075504843) q[9];
rz(pi/2) q[9];
cx q[8],q[9];
rz(0) q[8];
sx q[8];
rz(3.9552885744414663) q[8];
sx q[8];
rz(3*pi) q[8];
rz(0) q[9];
sx q[9];
rz(2.32789673273812) q[9];
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
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[1],q[8];
rz(4.960353461701102) q[8];
cx q[1],q[8];
cx q[14],q[1];
rz(-0.8556985182586128) q[1];
cx q[14],q[1];
rz(0.8556985182586128) q[1];
cx q[14],q[1];
rz(pi/2) q[1];
rz(3.059147140361007) q[14];
cx q[14],q[4];
rz(0) q[4];
sx q[4];
rz(2.226597779263985) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[14],q[4];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[4];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-0.2982546459121749) q[8];
sx q[8];
rz(6.617830511835823) q[8];
rz(-pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-pi/2) q[9];
rz(6.069096075504843) q[9];
rz(pi/2) q[9];
cx q[9],q[12];
x q[9];
rz(0) q[9];
sx q[9];
rz(5.638331155956776) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[12],q[9];
rz(0) q[9];
sx q[9];
rz(0.6448541512228099) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[12],q[9];
rz(2.387115078500571) q[9];
cx q[9],q[12];
rz(-2.387115078500571) q[12];
cx q[9],q[12];
rz(2.387115078500571) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[0],q[12];
rz(-pi/4) q[12];
cx q[0],q[12];
cx q[0],q[13];
rz(pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[1];
cx q[1],q[12];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(4.876543847111038) q[12];
rz(pi/2) q[12];
rz(5.99952732909904) q[13];
cx q[0],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(1.7111871495558115) q[13];
cx q[16],q[13];
rz(-4.261934324555408) q[13];
sx q[13];
rz(0.1595658240251452) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[16],q[13];
rz(0) q[13];
sx q[13];
rz(6.123619483154441) q[13];
sx q[13];
rz(11.975525135768976) q[13];
rz(0) q[13];
sx q[13];
rz(5.031469009453037) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[15],q[13];
rz(0) q[13];
sx q[13];
rz(1.2517162977265492) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[15],q[13];
rz(1.573205814615371) q[15];
cx q[16],q[2];
rz(pi/4) q[16];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[16],q[2];
rz(-pi/4) q[2];
cx q[16],q[2];
rz(4.718684904248475) q[16];
sx q[16];
rz(6.983672527966328) q[16];
sx q[16];
rz(9.684615321433984) q[16];
id q[16];
rz(-pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[2],q[15];
rz(-1.573205814615371) q[15];
cx q[2],q[15];
rz(pi) q[15];
x q[15];
rz(pi/2) q[2];
rz(0.18927432469684247) q[2];
cx q[3],q[1];
cx q[1],q[3];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
x q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[13],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi) q[1];
x q[1];
rz(-pi/4) q[1];
rz(-pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[3];
rz(1.4593531533858972) q[9];
rz(pi/2) q[9];
cx q[8],q[9];
rz(0) q[8];
sx q[8];
rz(0.6605751707638174) q[8];
sx q[8];
rz(3*pi) q[8];
rz(0) q[9];
sx q[9];
rz(0.6605751707638174) q[9];
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
cx q[8],q[6];
rz(3.714187147669167) q[6];
cx q[6],q[11];
rz(pi/2) q[11];
rz(pi/4) q[11];
cx q[11],q[17];
rz(-pi/4) q[17];
cx q[11],q[17];
rz(pi) q[11];
rz(5.1981776700331) q[11];
cx q[11],q[2];
rz(pi/4) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(-5.1981776700331) q[2];
sx q[2];
rz(2.6230399618487428) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[11],q[2];
rz(0) q[2];
sx q[2];
rz(3.6601453453308435) q[2];
sx q[2];
rz(14.433681306105637) q[2];
rz(-4.965301563712058) q[6];
rz(pi/2) q[6];
x q[8];
rz(5.609232780379475) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[14];
rz(3.2599432436071094) q[14];
cx q[8],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[6];
rz(0) q[14];
sx q[14];
rz(3.7375933974735522) q[14];
sx q[14];
rz(3*pi) q[14];
rz(0) q[6];
sx q[6];
rz(2.545591909706034) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[14],q[6];
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
rz(-pi/2) q[6];
rz(4.965301563712058) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[6],q[14];
rz(3.464538810120762) q[14];
cx q[6],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/2) q[14];
rz(3.047839710753825) q[14];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-pi/2) q[9];
rz(-1.4593531533858972) q[9];
rz(-0.8932731359001265) q[9];
cx q[5],q[9];
rz(-1.1830564220796815) q[9];
sx q[9];
rz(1.0752909906308736) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[5],q[9];
rz(2.137256171733192) q[5];
cx q[5],q[0];
rz(-2.137256171733192) q[0];
cx q[5],q[0];
rz(2.137256171733192) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi) q[0];
rz(0.6810971280275048) q[0];
id q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(2.8081806872039223) q[5];
cx q[4],q[5];
rz(-2.8081806872039223) q[5];
cx q[4],q[5];
rz(1.9815033096291563) q[4];
rz(pi/2) q[4];
cx q[4],q[14];
rz(-3.047839710753825) q[14];
cx q[4],q[14];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[17],q[5];
rz(0.0174084147669649) q[5];
cx q[17],q[5];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-2.1602463223473327) q[5];
rz(pi/2) q[5];
cx q[16],q[5];
rz(0) q[16];
sx q[16];
rz(5.887839486270243) q[16];
sx q[16];
rz(3*pi) q[16];
rz(0) q[5];
sx q[5];
rz(0.39534582090934256) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[16],q[5];
rz(-pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(2.5899739471810808) q[16];
rz(-pi/2) q[5];
rz(2.1602463223473327) q[5];
rz(pi/2) q[5];
sx q[5];
rz(4.757037804887152) q[5];
sx q[5];
rz(5*pi/2) q[5];
rz(0) q[9];
sx q[9];
rz(5.207894316548712) q[9];
sx q[9];
rz(11.501107518749187) q[9];
rz(-pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[12];
rz(0) q[12];
sx q[12];
rz(2.0825642090230154) q[12];
sx q[12];
rz(3*pi) q[12];
rz(0) q[9];
sx q[9];
rz(2.0825642090230154) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[9],q[12];
rz(-pi/2) q[12];
rz(-4.876543847111038) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[3];
cx q[3],q[12];
rz(1.7232226163665036) q[12];
cx q[12],q[8];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-1.7232226163665036) q[8];
cx q[12],q[8];
cx q[12],q[7];
rz(5.9636952219242305) q[7];
cx q[12],q[7];
rz(0) q[12];
sx q[12];
rz(3.6108719662229456) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[17],q[12];
rz(0) q[12];
sx q[12];
rz(2.6723133409566406) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[17],q[12];
rz(5.590651411721145) q[12];
sx q[12];
rz(7.692228994226141) q[12];
sx q[12];
rz(10.60760410477153) q[12];
cx q[17],q[6];
cx q[6],q[17];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(1.7232226163665036) q[8];
cx q[8],q[0];
rz(-0.6810971280275048) q[0];
cx q[8],q[0];
rz(2.930178381913837) q[0];
rz(0) q[0];
sx q[0];
rz(8.94731755943441) q[0];
sx q[0];
rz(3*pi) q[0];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/4) q[8];
cx q[7],q[8];
rz(-pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[1],q[7];
rz(2.7517559289557108) q[7];
cx q[1],q[7];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(-pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-pi/2) q[8];
rz(1.720926670629984) q[8];
cx q[8],q[2];
rz(-1.720926670629984) q[2];
cx q[8],q[2];
rz(1.720926670629984) q[2];
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
cx q[10],q[9];
rz(2.922840589071591) q[9];
cx q[10],q[9];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(4.607963923688153) q[10];
rz(pi/4) q[10];
cx q[10],q[13];
rz(-pi/4) q[13];
cx q[10],q[13];
rz(2.1911263420121356) q[10];
sx q[10];
rz(4.988494850039027) q[10];
sx q[10];
rz(14.962297670950786) q[10];
rz(pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(-pi/2) q[9];
cx q[3],q[9];
rz(pi/2) q[3];
cx q[3],q[13];
cx q[13],q[3];
cx q[11],q[13];
cx q[13],q[11];
cx q[11],q[13];
rz(-pi/2) q[3];
rz(pi/2) q[9];
id q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(0.2989898119742936) q[9];
cx q[15],q[9];
rz(-0.2989898119742936) q[9];
cx q[15],q[9];
sx q[15];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[10],q[9];
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