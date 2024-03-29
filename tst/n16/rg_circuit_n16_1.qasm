OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
creg c[16];
rz(1.129406185874691) q[0];
sx q[0];
rz(7.5975498261285965) q[0];
sx q[0];
rz(8.295371774894688) q[0];
x q[0];
sx q[1];
rz(-0.8371058622409595) q[4];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(4.28940693574336) q[6];
sx q[6];
rz(3.2448238394947495) q[6];
cx q[6],q[1];
rz(3.8439892204249855) q[1];
cx q[6],q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(0) q[1];
sx q[1];
rz(5.715698718294045) q[1];
sx q[1];
rz(3*pi) q[1];
id q[6];
sx q[6];
rz(pi) q[6];
rz(pi/2) q[6];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[8],q[2];
cx q[2],q[8];
rz(1.6042010053697429) q[2];
rz(0) q[2];
sx q[2];
rz(4.446248697415266) q[2];
sx q[2];
rz(3*pi) q[2];
rz(0.4947067836917049) q[2];
rz(pi) q[2];
rz(-2.0650099052097426) q[2];
rz(-2.397911035367838) q[8];
sx q[8];
rz(3.8197076463248028) q[8];
sx q[8];
rz(11.822688996137217) q[8];
rz(-pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-pi/2) q[9];
rz(pi/4) q[9];
id q[10];
sx q[11];
rz(2.903266102422312) q[11];
rz(2.936785128257654) q[11];
rz(1.7616172441067242) q[12];
cx q[12],q[4];
rz(-1.7616172441067242) q[4];
sx q[4];
rz(1.2840084134113554) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[12],q[4];
rz(1.0281480523133562) q[12];
rz(3.5136749626182087) q[12];
rz(0) q[4];
sx q[4];
rz(4.999176893768231) q[4];
sx q[4];
rz(12.023501067117063) q[4];
rz(-pi/4) q[4];
rz(-pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[8],q[4];
rz(3.9006481372239974) q[4];
cx q[8],q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(0.27774005770843235) q[4];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(0.614923956090921) q[8];
rz(pi) q[8];
x q[8];
rz(pi/2) q[13];
cx q[5],q[13];
cx q[13],q[5];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(0.38504432913987396) q[13];
cx q[11],q[13];
rz(-2.936785128257654) q[13];
sx q[13];
rz(0.20846544015384572) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[11],q[13];
rz(3.761620588430615) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(0) q[13];
sx q[13];
rz(6.07471986702574) q[13];
sx q[13];
rz(11.97651875988716) q[13];
rz(2.921351844210984) q[13];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[9],q[5];
rz(-pi/4) q[5];
cx q[9],q[5];
rz(pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/4) q[9];
rz(pi/4) q[14];
cx q[14],q[7];
rz(-pi/4) q[7];
cx q[14],q[7];
rz(1.989339424431431) q[14];
cx q[12],q[14];
rz(-3.5136749626182087) q[14];
sx q[14];
rz(1.9417276334452545) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[12],q[14];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(0) q[14];
sx q[14];
rz(4.341457673734332) q[14];
sx q[14];
rz(10.949113498956157) q[14];
rz(1.998360766700326) q[14];
cx q[0],q[14];
rz(-1.998360766700326) q[14];
cx q[0],q[14];
rz(4.927992868461429) q[0];
cx q[0],q[4];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/4) q[14];
rz(-4.927992868461429) q[4];
sx q[4];
rz(2.8961124731356023) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[0],q[4];
rz(1.5053310017042563) q[0];
rz(0) q[4];
sx q[4];
rz(3.387072834043984) q[4];
sx q[4];
rz(14.075030771522375) q[4];
rz(pi/4) q[4];
rz(pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(0) q[7];
sx q[7];
rz(4.988060386910181) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[9],q[12];
rz(-pi/4) q[12];
cx q[9],q[12];
rz(pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(0) q[9];
sx q[9];
rz(5.2517303434583535) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[12],q[9];
rz(0) q[9];
sx q[9];
rz(1.0314549637212331) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[12],q[9];
rz(1.3514200034442183) q[12];
rz(4.270047180423993) q[12];
rz(1.6591525832318936) q[12];
rz(3.7104499143297556) q[12];
rz(4.468428026278568) q[9];
sx q[9];
rz(3.329512639148353) q[9];
rz(-5.249969729167648) q[9];
rz(pi/2) q[9];
cx q[3],q[15];
rz(3.991034872027401) q[15];
cx q[3],q[15];
cx q[10],q[3];
cx q[15],q[7];
rz(4.426857785163906) q[3];
cx q[10],q[3];
cx q[13],q[10];
rz(-2.921351844210984) q[10];
cx q[13],q[10];
rz(2.921351844210984) q[10];
rz(pi) q[10];
rz(2.6894697506186547) q[10];
cx q[13],q[1];
rz(0) q[1];
sx q[1];
rz(0.5674865888855409) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[13],q[1];
rz(pi/2) q[1];
rz(0) q[1];
sx q[1];
rz(3.465086564883867) q[1];
sx q[1];
rz(3*pi) q[1];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(-3.2545505479655326) q[3];
sx q[3];
rz(3.7086399081676804) q[3];
sx q[3];
rz(12.679328508734912) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
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
rz(0.7399356598554114) q[3];
sx q[3];
rz(7.9051370120123226) q[3];
sx q[3];
rz(8.684842300913967) q[3];
rz(2.6102566430859975) q[3];
cx q[4],q[1];
rz(0) q[1];
sx q[1];
rz(2.818098742295719) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[4],q[1];
rz(2.3791730691972885) q[1];
cx q[12],q[1];
rz(-3.7104499143297556) q[1];
sx q[1];
rz(1.4279729103597758) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[12],q[1];
rz(0) q[1];
sx q[1];
rz(4.855212396819811) q[1];
sx q[1];
rz(10.756054805901847) q[1];
rz(pi/2) q[1];
rz(6.25051538661841) q[12];
rz(2.437318664556638) q[12];
rz(pi/4) q[4];
rz(0) q[7];
sx q[7];
rz(1.2951249202694046) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[15],q[7];
rz(2.8865400304935567) q[15];
cx q[15],q[5];
rz(-2.8865400304935567) q[5];
cx q[15],q[5];
rz(pi/4) q[15];
cx q[15],q[11];
rz(-pi/4) q[11];
cx q[15],q[11];
rz(pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[0],q[11];
rz(-1.5053310017042563) q[11];
cx q[0],q[11];
rz(3.68761385069599) q[0];
cx q[0],q[2];
rz(1.5053310017042563) q[11];
rz(1.885157637501492) q[11];
sx q[11];
rz(4.161836180397741) q[11];
sx q[11];
rz(12.393867615834864) q[11];
id q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi) q[15];
x q[15];
rz(pi/4) q[15];
rz(-3.68761385069599) q[2];
sx q[2];
rz(0.8681988465206141) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[0],q[2];
rz(0.7189912531388292) q[0];
sx q[0];
rz(6.682780250171142) q[0];
sx q[0];
rz(14.454337337607825) q[0];
rz(1.1679458075974802) q[0];
rz(0) q[2];
sx q[2];
rz(5.414986460658972) q[2];
sx q[2];
rz(15.177401716675112) q[2];
rz(2.8865400304935567) q[5];
rz(0.31513637296847513) q[5];
rz(0.576377265315279) q[5];
sx q[5];
rz(4.995949206135078) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[5],q[6];
cx q[6],q[5];
rz(-pi/4) q[5];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[7];
cx q[7],q[14];
rz(-pi/4) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/2) q[14];
cx q[14],q[10];
rz(-2.6894697506186547) q[10];
cx q[14],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[9];
rz(0) q[14];
sx q[14];
rz(5.42129886609589) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[15],q[10];
rz(-pi/4) q[10];
cx q[15],q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(5.331624338403551) q[15];
cx q[0],q[15];
rz(-1.1679458075974802) q[15];
cx q[0],q[15];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(1.1679458075974802) q[15];
cx q[5],q[10];
rz(2.6014946049898042) q[10];
cx q[5],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(0) q[5];
sx q[5];
rz(6.480190619143978) q[5];
sx q[5];
rz(3*pi) q[5];
rz(-pi/2) q[5];
cx q[10],q[5];
rz(pi/2) q[5];
id q[5];
rz(-pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[7];
sx q[7];
rz(6.924068918571095) q[7];
sx q[7];
rz(5*pi/2) q[7];
rz(0) q[7];
sx q[7];
rz(3.6657001329055374) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[6],q[7];
cx q[7],q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[8],q[13];
cx q[13],q[8];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/4) q[13];
cx q[13],q[11];
rz(-pi/4) q[11];
cx q[13],q[11];
cx q[1],q[13];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
id q[11];
rz(5.595192541575421) q[11];
rz(pi/2) q[11];
rz(pi) q[11];
x q[11];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[1],q[13];
rz(2.1859072450315) q[13];
cx q[1],q[13];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(1.6592935980005485) q[8];
cx q[2],q[8];
rz(-1.6592935980005485) q[8];
cx q[2],q[8];
rz(pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[6];
rz(5.026125568509123) q[6];
cx q[8],q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[6],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/4) q[6];
cx q[6],q[2];
rz(-pi/4) q[2];
cx q[6],q[2];
rz(pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[15],q[8];
cx q[8],q[15];
cx q[15],q[8];
rz(pi/2) q[15];
cx q[1],q[15];
rz(1.5155730285028832) q[15];
cx q[1],q[15];
rz(5.446042859765848) q[15];
rz(pi/2) q[15];
cx q[5],q[15];
rz(0) q[15];
sx q[15];
rz(3.1310363066429536) q[15];
sx q[15];
rz(3*pi) q[15];
rz(0) q[5];
sx q[5];
rz(3.1310363066429536) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[5],q[15];
rz(-pi/2) q[15];
rz(-5.446042859765848) q[15];
rz(-pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(0) q[9];
sx q[9];
rz(0.8618864410836959) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[14],q[9];
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
cx q[4],q[14];
rz(-pi/4) q[14];
cx q[4],q[14];
rz(pi/4) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(5.569470098690069) q[14];
sx q[14];
rz(9.241677784176165) q[14];
sx q[14];
rz(12.212308182039735) q[14];
rz(0) q[14];
sx q[14];
rz(8.049128953889758) q[14];
sx q[14];
rz(3*pi) q[14];
rz(1.934031631220274) q[4];
cx q[12],q[4];
rz(-2.437318664556638) q[4];
sx q[4];
rz(1.159861147625108) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[12],q[4];
rz(-1.8141535154449446) q[12];
sx q[12];
rz(5.138433576139318) q[12];
sx q[12];
rz(11.238931476214324) q[12];
id q[12];
id q[12];
rz(0.5813778280663668) q[12];
rz(0) q[4];
sx q[4];
rz(5.123324159554478) q[4];
sx q[4];
rz(9.928064994105743) q[4];
id q[4];
rz(pi/2) q[4];
cx q[6],q[4];
cx q[4],q[6];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(2.449068937658491) q[4];
sx q[4];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(1.0480693771695058) q[6];
cx q[10],q[6];
rz(-1.0480693771695058) q[6];
cx q[10],q[6];
rz(pi/2) q[10];
rz(-pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
cx q[6],q[4];
rz(-0.7310891006366064) q[4];
sx q[4];
rz(2.744303975742171) q[4];
x q[6];
rz(-pi/2) q[9];
rz(5.249969729167648) q[9];
rz(1.6065083817478225) q[9];
rz(2.677596407702489) q[9];
cx q[9],q[3];
rz(-2.677596407702489) q[3];
sx q[3];
rz(2.60942207930589) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[9],q[3];
rz(0) q[3];
sx q[3];
rz(3.673763227873696) q[3];
sx q[3];
rz(9.49211772538587) q[3];
rz(pi/2) q[3];
rz(5.972640572189383) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(1.6649916841055246) q[3];
sx q[3];
rz(8.663766360012833) q[3];
sx q[3];
rz(11.286632914500483) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[1],q[3];
rz(pi/4) q[1];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[1],q[3];
rz(-pi/4) q[3];
cx q[1],q[3];
rz(2.1087474298818587) q[1];
rz(pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[10],q[3];
rz(4.4518134088421855) q[3];
cx q[10],q[3];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi) q[3];
x q[3];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/4) q[9];
cx q[7],q[9];
rz(pi/2) q[7];
cx q[0],q[7];
cx q[7],q[0];
cx q[14],q[0];
rz(0.27659590070553536) q[0];
cx q[14],q[0];
rz(-0.27659590070553536) q[0];
cx q[14],q[0];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[2];
rz(4.710173931127632) q[2];
cx q[7],q[2];
rz(pi/2) q[2];
sx q[2];
rz(3.3047011212586703) q[2];
sx q[2];
rz(5*pi/2) q[2];
rz(4.09580466979207) q[2];
rz(3.695537124928689) q[2];
cx q[2],q[12];
rz(-3.695537124928689) q[12];
sx q[12];
rz(1.4081146461494296) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[2],q[12];
rz(0) q[12];
sx q[12];
rz(4.875070661030157) q[12];
sx q[12];
rz(12.538937257631702) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(1.229365670404627) q[12];
rz(5.479074813858444) q[12];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[5],q[2];
rz(2.09092527514416) q[2];
cx q[5],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/4) q[5];
rz(-pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[13];
rz(0.14873230933053683) q[13];
cx q[7],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(-pi/2) q[13];
cx q[0],q[13];
cx q[0],q[11];
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
rz(0.2376892360103704) q[11];
cx q[12],q[11];
rz(-5.479074813858444) q[11];
sx q[11];
rz(2.9938577072914003) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[12],q[11];
rz(0) q[11];
sx q[11];
rz(3.289327599888186) q[11];
sx q[11];
rz(14.666163538617454) q[11];
rz(-pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[15],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/4) q[15];
cx q[15],q[13];
rz(-pi/4) q[13];
cx q[15],q[13];
rz(pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[10],q[13];
rz(-pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[11],q[10];
rz(2.635505722785626) q[10];
cx q[11],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(0.5089546106645226) q[10];
sx q[10];
rz(5.656263423327231) q[10];
sx q[10];
rz(11.727905379714697) q[10];
rz(-pi/4) q[10];
rz(4.084727104856796) q[10];
rz(pi/2) q[10];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(-1.7241082548894473) q[11];
rz(pi/2) q[11];
rz(0.41347128922105464) q[13];
cx q[13],q[3];
rz(pi/2) q[15];
rz(-0.41347128922105464) q[3];
cx q[13],q[3];
rz(0.41347128922105464) q[3];
cx q[5],q[12];
cx q[12],q[5];
cx q[5],q[12];
rz(-4.57805802655175) q[12];
sx q[12];
rz(5.205424793574358) q[12];
sx q[12];
rz(14.002835987321129) q[12];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(0) q[7];
sx q[7];
rz(5.213100749938164) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[14],q[7];
rz(0) q[7];
sx q[7];
rz(1.070084557241422) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[14],q[7];
rz(pi) q[14];
cx q[14],q[4];
rz(pi/2) q[14];
cx q[7],q[1];
rz(-2.1087474298818587) q[1];
cx q[7],q[1];
sx q[1];
cx q[15],q[1];
rz(0) q[1];
sx q[1];
rz(5.2034677247192995) q[1];
sx q[1];
rz(3*pi) q[1];
sx q[1];
rz(2.0482778567542215) q[1];
cx q[1],q[0];
rz(-2.0482778567542215) q[0];
cx q[1],q[0];
rz(2.0482778567542215) q[0];
rz(pi/2) q[0];
sx q[0];
rz(4.078015942029538) q[0];
sx q[0];
rz(5*pi/2) q[0];
rz(1.111963657037589) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(4.76323315059426) q[1];
sx q[1];
rz(5.37318278196198) q[1];
sx q[1];
rz(13.198830877113291) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
x q[15];
rz(-pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(0) q[7];
sx q[7];
rz(5.309126831318063) q[7];
sx q[7];
rz(3*pi) q[7];
rz(pi) q[7];
rz(pi/4) q[7];
rz(-pi/4) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(-pi/2) q[9];
rz(pi/4) q[9];
cx q[9],q[8];
cx q[8],q[9];
cx q[9],q[8];
x q[8];
rz(pi) q[9];
cx q[8],q[9];
cx q[9],q[8];
rz(3.320417573098411) q[8];
cx q[9],q[6];
cx q[6],q[9];
cx q[6],q[8];
cx q[8],q[6];
cx q[6],q[8];
rz(3.5160488240251633) q[6];
rz(pi/2) q[6];
cx q[15],q[6];
rz(0) q[15];
sx q[15];
rz(1.7401226037962496) q[15];
sx q[15];
rz(3*pi) q[15];
rz(0) q[6];
sx q[6];
rz(1.7401226037962496) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[15],q[6];
rz(-pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(2.450487147072531) q[15];
cx q[5],q[15];
rz(-2.450487147072531) q[15];
cx q[5],q[15];
id q[15];
rz(4.632574997285742) q[15];
sx q[15];
rz(9.123075455144189) q[15];
sx q[15];
rz(11.485683327419515) q[15];
rz(pi) q[5];
x q[5];
rz(-pi/2) q[6];
rz(-3.5160488240251633) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/4) q[6];
cx q[13],q[6];
rz(-pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(-pi/2) q[6];
rz(2.9745681065696172) q[8];
cx q[8],q[3];
cx q[3],q[8];
cx q[8],q[3];
rz(-pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(1.4528957908247768) q[8];
cx q[6],q[8];
rz(-1.4528957908247768) q[8];
cx q[6],q[8];
rz(-pi/2) q[6];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(4.364966418334184) q[8];
sx q[8];
rz(5.701659565428114) q[8];
sx q[8];
rz(10.539545923644383) q[8];
rz(pi/2) q[8];
sx q[8];
rz(4.801453324239728) q[8];
sx q[8];
rz(5*pi/2) q[8];
rz(-pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(1.0064052414404219) q[9];
rz(pi/2) q[9];
cx q[2],q[9];
rz(0) q[2];
sx q[2];
rz(1.8199149342632766) q[2];
sx q[2];
rz(3*pi) q[2];
rz(0) q[9];
sx q[9];
rz(1.8199149342632766) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[2],q[9];
rz(-pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/4) q[2];
cx q[4],q[2];
rz(-pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
id q[4];
rz(-pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[13],q[4];
rz(4.849537872395141) q[4];
cx q[13],q[4];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(-2.527321971791828) q[13];
sx q[13];
rz(6.247362061831668) q[13];
sx q[13];
rz(11.952099932561207) q[13];
rz(-0.5509831016967954) q[13];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(-pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[4],q[10];
rz(0) q[10];
sx q[10];
rz(1.246080326009578) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[4];
sx q[4];
rz(1.246080326009578) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[4],q[10];
rz(-pi/2) q[10];
rz(-4.084727104856796) q[10];
rz(0) q[10];
sx q[10];
rz(6.667401531857687) q[10];
sx q[10];
rz(3*pi) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(2.032116930459579) q[4];
rz(0.9182042716734252) q[4];
sx q[4];
rz(5.802383936224322) q[4];
sx q[4];
rz(15.300903697802916) q[4];
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
rz(4.220909094722593) q[2];
rz(pi/2) q[2];
cx q[3],q[2];
rz(0) q[2];
sx q[2];
rz(0.6731635488323366) q[2];
sx q[2];
rz(3*pi) q[2];
rz(0) q[3];
sx q[3];
rz(0.6731635488323366) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[3],q[2];
rz(-pi/2) q[2];
rz(-4.220909094722593) q[2];
rz(2.9609250351951952) q[2];
rz(-pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
cx q[3],q[2];
rz(-2.9609250351951952) q[2];
cx q[3],q[2];
rz(3.8324552119042896) q[2];
rz(0.6393404784601269) q[3];
cx q[2],q[3];
rz(-3.8324552119042896) q[3];
sx q[3];
rz(0.0851780821575212) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[2],q[3];
rz(-1.9418280285768732) q[2];
rz(0) q[3];
sx q[3];
rz(6.1980072250220655) q[3];
sx q[3];
rz(12.617892694213541) q[3];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[7];
cx q[8],q[3];
rz(5.441177031708196) q[3];
cx q[8],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(-pi/2) q[9];
rz(-1.0064052414404219) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[14];
cx q[14],q[9];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[11];
rz(0) q[11];
sx q[11];
rz(1.7961499297952) q[11];
sx q[11];
rz(3*pi) q[11];
rz(0) q[14];
sx q[14];
rz(4.4870353773843865) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[14],q[11];
rz(-pi/2) q[11];
rz(1.7241082548894473) q[11];
rz(pi) q[11];
rz(0.7496299433275525) q[11];
sx q[11];
rz(7.593982332047455) q[11];
sx q[11];
rz(8.675148017441828) q[11];
rz(5.697973267584666) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
cx q[7],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(1.134024452332448) q[14];
x q[7];
cx q[7],q[5];
rz(3.1392847003673543) q[5];
rz(5.003847759557408) q[5];
cx q[5],q[13];
rz(-5.003847759557408) q[13];
sx q[13];
rz(2.6992204701900997) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[5],q[13];
rz(0) q[13];
sx q[13];
rz(3.5839648369894865) q[13];
sx q[13];
rz(14.979608822023582) q[13];
cx q[13],q[10];
rz(3.516244874318205) q[10];
cx q[13],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(2.3003989150417854) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(3.0398847879945707) q[9];
cx q[12],q[9];
rz(-3.0398847879945707) q[9];
cx q[12],q[9];
cx q[12],q[14];
rz(-1.134024452332448) q[14];
cx q[12],q[14];
cx q[12],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[12];
cx q[12],q[1];
rz(-pi/4) q[1];
cx q[12],q[1];
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
cx q[11],q[1];
rz(2.9278347204649093) q[1];
cx q[11],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(2.3982819327941396) q[12];
cx q[12],q[2];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-2.3982819327941396) q[2];
sx q[2];
rz(1.9682949683904984) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[12],q[2];
rz(0) q[2];
sx q[2];
rz(4.314890338789088) q[2];
sx q[2];
rz(13.764887922140392) q[2];
cx q[7],q[14];
cx q[14],q[7];
cx q[7],q[14];
rz(pi/4) q[14];
rz(5.823322971885502) q[7];
sx q[7];
rz(3.7926238509299486) q[7];
sx q[7];
rz(14.25946990733663) q[7];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[6];
rz(pi/2) q[6];
cx q[6],q[15];
cx q[15],q[6];
rz(pi/2) q[15];
sx q[15];
rz(5.7475939028241445) q[15];
sx q[15];
rz(5*pi/2) q[15];
rz(pi) q[6];
x q[6];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
cx q[9],q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
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
