OPENQASM 2.0;
include "qelib1.inc";
qreg q[19];
creg c[19];
rz(1.0561169796342562) q[0];
rz(4.418379522910169) q[2];
sx q[2];
rz(5.154535049435655) q[2];
rz(0) q[2];
sx q[2];
rz(7.072837625371662) q[2];
sx q[2];
rz(3*pi) q[2];
rz(-pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[1],q[4];
rz(pi) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(1.4027670339123173) q[4];
sx q[4];
rz(7.479110380999616) q[4];
rz(-pi/2) q[4];
rz(-pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[6];
rz(1.0218358790820194) q[7];
rz(5.498630483288284) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
x q[9];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(0.0018018400837895958) q[10];
rz(pi/2) q[11];
sx q[11];
rz(6.934925497587328) q[11];
sx q[11];
rz(5*pi/2) q[11];
rz(2.2506731212571354) q[11];
rz(-pi/2) q[11];
rz(3.7549788788623775) q[12];
rz(pi/2) q[12];
cx q[3],q[12];
rz(0) q[12];
sx q[12];
rz(1.9664051544222139) q[12];
sx q[12];
rz(3*pi) q[12];
rz(0) q[3];
sx q[3];
rz(1.9664051544222139) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[3],q[12];
rz(-pi/2) q[12];
rz(-3.7549788788623775) q[12];
rz(0) q[12];
sx q[12];
rz(5.084847042662871) q[12];
sx q[12];
rz(3*pi) q[12];
rz(-pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(-1.5707740231451939) q[3];
sx q[3];
rz(4.034775935104328) q[3];
sx q[3];
rz(10.995551983914574) q[3];
rz(pi) q[3];
x q[3];
rz(pi/2) q[3];
sx q[3];
rz(3.1439958631181217) q[3];
sx q[3];
rz(5*pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(9.4156233407718) q[3];
sx q[3];
rz(5*pi/2) q[3];
cx q[9],q[12];
rz(0) q[12];
sx q[12];
rz(1.1983382645167147) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[9],q[12];
rz(pi/4) q[12];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[9],q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-6.2142838139757295) q[9];
rz(pi/2) q[9];
cx q[4],q[9];
rz(0) q[4];
sx q[4];
rz(3.221282331291405) q[4];
sx q[4];
rz(3*pi) q[4];
rz(0) q[9];
sx q[9];
rz(3.0619029758881813) q[9];
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
x q[4];
rz(-0.4050039945552326) q[4];
rz(-pi/2) q[9];
rz(6.2142838139757295) q[9];
sx q[13];
cx q[6],q[13];
rz(pi/2) q[13];
sx q[13];
rz(7.573658178677627) q[13];
sx q[13];
rz(5*pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/4) q[13];
x q[6];
rz(-pi/2) q[14];
rz(2.014656426543506) q[14];
rz(pi/4) q[15];
cx q[15],q[10];
rz(-0.0018018400837895958) q[10];
cx q[15],q[10];
rz(-pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(2.035418804956984) q[15];
rz(pi/2) q[15];
cx q[10],q[15];
rz(0) q[10];
sx q[10];
rz(2.8911997850683546) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[15];
sx q[15];
rz(2.8911997850683546) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[10],q[15];
rz(-pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
id q[10];
rz(pi/4) q[10];
rz(-pi/2) q[15];
rz(-2.035418804956984) q[15];
rz(5.045157350557961) q[15];
cx q[0],q[16];
rz(-1.0561169796342562) q[16];
cx q[0],q[16];
rz(1.1705227615307063) q[0];
rz(1.0561169796342562) q[16];
cx q[0],q[16];
rz(-1.1705227615307063) q[16];
cx q[0],q[16];
rz(pi/4) q[0];
cx q[0],q[1];
rz(-pi/4) q[1];
cx q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[11],q[0];
rz(4.335356107574907) q[0];
cx q[11],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
id q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(1.8624573974350396) q[11];
sx q[11];
rz(4.473664056516103) q[11];
rz(1.1705227615307063) q[16];
rz(2.086284849844498) q[16];
cx q[3],q[11];
rz(3.1764092837816906) q[11];
rz(0) q[3];
sx q[3];
rz(3.396337000265115) q[3];
sx q[3];
rz(3*pi) q[3];
rz(pi/4) q[17];
cx q[17],q[5];
rz(-pi/4) q[5];
cx q[17],q[5];
cx q[17],q[6];
rz(pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/4) q[5];
rz(pi/2) q[5];
cx q[6],q[17];
rz(-0.22977644821443066) q[17];
sx q[17];
rz(7.517905411907988) q[17];
sx q[17];
rz(9.65455440898381) q[17];
rz(-3.665803663160646) q[17];
sx q[17];
rz(5.975215246120795) q[17];
sx q[17];
rz(13.090581623930024) q[17];
id q[17];
rz(pi) q[17];
rz(3.027371636951157) q[17];
cx q[6],q[13];
rz(-pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-pi/2) q[13];
rz(0.8853526634016737) q[13];
cx q[15],q[13];
rz(-5.045157350557961) q[13];
sx q[13];
rz(0.2396947005673451) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[15],q[13];
rz(0) q[13];
sx q[13];
rz(6.043490606612242) q[13];
sx q[13];
rz(13.584582647925666) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(4.095315786466438) q[13];
cx q[13],q[4];
rz(pi/2) q[15];
rz(-0.5493859775526784) q[15];
rz(-4.095315786466438) q[4];
sx q[4];
rz(0.9509984165977001) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[13],q[4];
rz(0) q[4];
sx q[4];
rz(5.332186890581886) q[4];
sx q[4];
rz(13.925097741791049) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(1.50472312061509) q[6];
cx q[7],q[5];
cx q[5],q[7];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(0) q[5];
sx q[5];
rz(5.341874539028332) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[1],q[5];
rz(0) q[5];
sx q[5];
rz(0.9413107681512543) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[1],q[5];
cx q[5],q[9];
rz(-1.1639877407249137) q[7];
rz(pi/2) q[7];
cx q[12],q[7];
rz(0) q[12];
sx q[12];
rz(6.258244280232273) q[12];
sx q[12];
rz(3*pi) q[12];
rz(0) q[7];
sx q[7];
rz(0.024941026947312928) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[12],q[7];
rz(-pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(-pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi/2) q[7];
rz(1.1639877407249137) q[7];
rz(-pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[12];
rz(0.2823840099092696) q[12];
cx q[7],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(5.544620491824668) q[12];
sx q[12];
rz(7.499067546713971) q[12];
sx q[12];
rz(11.436616749043047) q[12];
rz(0) q[12];
sx q[12];
rz(3.304433681820705) q[12];
sx q[12];
rz(3*pi) q[12];
sx q[12];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(1.7916592924933168) q[7];
rz(2.6769003552564086) q[7];
cx q[7],q[15];
rz(-2.6769003552564086) q[15];
sx q[15];
rz(1.5463495447419486) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[7],q[15];
rz(0) q[15];
sx q[15];
rz(4.736835762437638) q[15];
sx q[15];
rz(12.651064293578466) q[15];
rz(pi) q[15];
rz(-pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[7],q[17];
rz(-3.027371636951157) q[17];
cx q[7],q[17];
rz(3.7741152436716803) q[17];
rz(pi/2) q[17];
rz(-pi/2) q[7];
cx q[9],q[5];
cx q[5],q[9];
rz(0) q[5];
sx q[5];
rz(5.882743753936213) q[5];
sx q[5];
rz(3*pi) q[5];
rz(1.5586826419641615) q[9];
rz(pi/4) q[9];
cx q[9],q[4];
rz(-pi/4) q[4];
cx q[9],q[4];
rz(pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(4.620334241242148) q[4];
sx q[4];
rz(7.577404197160703) q[4];
sx q[4];
rz(10.659475070535605) q[4];
rz(5.101507035678267) q[4];
rz(0) q[4];
sx q[4];
rz(5.26986080345436) q[4];
sx q[4];
rz(3*pi) q[4];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/4) q[18];
cx q[8],q[18];
rz(-pi/4) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi/2) q[18];
rz(1.6544532770732527) q[18];
cx q[16],q[18];
rz(-2.086284849844498) q[18];
cx q[16],q[18];
rz(2.086284849844498) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[6],q[16];
rz(-1.50472312061509) q[16];
cx q[6],q[16];
rz(1.50472312061509) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(2.284698400731661) q[6];
cx q[1],q[6];
rz(-2.284698400731661) q[6];
cx q[1],q[6];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(0.06744566878116132) q[1];
cx q[11],q[1];
rz(-3.1764092837816906) q[1];
sx q[1];
rz(1.4965947285264931) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[11],q[1];
rz(0) q[1];
sx q[1];
rz(4.786590578653093) q[1];
sx q[1];
rz(12.533741575769909) q[1];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(2.295998754714861) q[6];
rz(0.9165996889103003) q[8];
cx q[14],q[8];
rz(-2.014656426543506) q[8];
sx q[8];
rz(0.9336221183104914) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[14],q[8];
cx q[2],q[14];
rz(1.6365054341127767) q[14];
cx q[2],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/4) q[2];
cx q[2],q[14];
rz(-pi/4) q[14];
cx q[2],q[14];
rz(pi/4) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
cx q[18],q[14];
cx q[14],q[18];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(0.8118015486996989) q[14];
rz(0.6827861252835935) q[18];
cx q[2],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/4) q[2];
cx q[2],q[16];
rz(-pi/4) q[16];
cx q[2],q[16];
rz(pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
cx q[16],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/4) q[16];
cx q[16],q[0];
rz(-pi/4) q[0];
cx q[16],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[15];
rz(2.6917333683096) q[15];
cx q[0],q[15];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-4.665872121902831) q[0];
sx q[0];
rz(6.492710211004454) q[0];
sx q[0];
rz(14.09065008267221) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(4.569067001505745) q[15];
rz(pi/2) q[15];
rz(pi/2) q[16];
cx q[16],q[12];
rz(1.5245474471145395) q[12];
x q[16];
cx q[2],q[5];
rz(0) q[5];
sx q[5];
rz(0.40044155324337405) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[2],q[5];
id q[2];
cx q[2],q[7];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(-1.6175921333249297) q[2];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/4) q[5];
cx q[6],q[18];
rz(-2.295998754714861) q[18];
sx q[18];
rz(0.10744238396177064) q[18];
sx q[18];
rz(3*pi) q[18];
cx q[6],q[18];
rz(0) q[18];
sx q[18];
rz(6.175742923217816) q[18];
sx q[18];
rz(11.037990590200646) q[18];
cx q[18],q[5];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[5];
cx q[5],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi) q[6];
x q[6];
cx q[6],q[1];
cx q[1],q[6];
cx q[6],q[1];
rz(pi/2) q[6];
rz(pi/2) q[7];
cx q[11],q[7];
cx q[7],q[11];
cx q[11],q[12];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[12];
cx q[11],q[12];
cx q[12],q[11];
x q[11];
rz(0) q[11];
sx q[11];
rz(6.0196460100039815) q[11];
sx q[11];
rz(3*pi) q[11];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(0) q[8];
sx q[8];
rz(5.349563188869094) q[8];
sx q[8];
rz(10.522834698402585) q[8];
rz(pi) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[10],q[8];
rz(-pi/4) q[8];
cx q[10],q[8];
cx q[14],q[10];
rz(-0.8118015486996989) q[10];
cx q[14],q[10];
rz(0.8118015486996989) q[10];
sx q[10];
rz(-pi/2) q[10];
rz(1.9268715499766644) q[10];
cx q[10],q[16];
cx q[14],q[3];
rz(-1.9268715499766644) q[16];
cx q[10],q[16];
rz(pi/2) q[10];
sx q[10];
rz(7.848574875264466) q[10];
sx q[10];
rz(5*pi/2) q[10];
rz(-pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(1.9268715499766644) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/4) q[16];
rz(0) q[3];
sx q[3];
rz(2.8868483069144713) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[14],q[3];
rz(-pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(2.1083754281033125) q[14];
rz(1.3790253791336597) q[3];
cx q[3],q[9];
cx q[5],q[14];
rz(-2.1083754281033125) q[14];
cx q[5],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[4];
rz(0) q[4];
sx q[4];
rz(1.0133245037252263) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[14],q[4];
rz(0) q[14];
sx q[14];
rz(3.22720335837353) q[14];
sx q[14];
rz(3*pi) q[14];
rz(2.7240765122104316) q[5];
rz(-0.24362667731552534) q[5];
sx q[5];
rz(7.143939464995683) q[5];
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[13],q[8];
cx q[8],q[13];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[17];
rz(0) q[13];
sx q[13];
rz(1.6387256593383437) q[13];
sx q[13];
rz(3*pi) q[13];
rz(0) q[17];
sx q[17];
rz(1.6387256593383437) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[13],q[17];
rz(-pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(-pi/2) q[17];
rz(-3.7741152436716803) q[17];
sx q[17];
cx q[6],q[17];
rz(0.6786394382692905) q[17];
sx q[17];
rz(3.96862726710596) q[17];
sx q[17];
rz(14.999044655212625) q[17];
rz(5.824458801584867) q[17];
rz(2.3308170705659683) q[17];
cx q[17],q[2];
rz(-2.3308170705659683) q[2];
sx q[2];
rz(1.7747296760036344) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[17],q[2];
rz(0) q[2];
sx q[2];
rz(4.508455631175952) q[2];
sx q[2];
rz(13.373187164660278) q[2];
id q[2];
rz(6.199851136232317) q[2];
sx q[2];
rz(4.989334105998825) q[2];
x q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[18];
rz(0.9272843691327496) q[18];
cx q[8],q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi) q[18];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[18],q[15];
rz(0) q[15];
sx q[15];
rz(1.382985843682818) q[15];
sx q[15];
rz(3*pi) q[15];
rz(0) q[18];
sx q[18];
rz(1.382985843682818) q[18];
sx q[18];
rz(3*pi) q[18];
cx q[18],q[15];
rz(-pi/2) q[15];
rz(-4.569067001505745) q[15];
rz(-pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
cx q[18],q[15];
cx q[15],q[18];
rz(2.896406417816389) q[18];
cx q[5],q[15];
rz(3.099950841881695) q[15];
cx q[5],q[15];
sx q[15];
rz(0.7994756191659416) q[5];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(2.380040912767499) q[8];
sx q[8];
rz(2.905376974407463) q[8];
rz(-pi/2) q[8];
cx q[7],q[8];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[4],q[7];
rz(0.48261088381260825) q[7];
cx q[4],q[7];
rz(0.2712367322137706) q[4];
cx q[4],q[17];
rz(-0.2712367322137706) q[17];
cx q[4],q[17];
rz(0.2712367322137706) q[17];
cx q[17],q[11];
rz(0) q[11];
sx q[11];
rz(0.2635392971756043) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[17],q[11];
rz(pi/2) q[17];
id q[4];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[8];
rz(-2.8592890227932743) q[8];
rz(pi/2) q[8];
cx q[10],q[8];
rz(0) q[10];
sx q[10];
rz(5.369457057263625) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[8];
sx q[8];
rz(0.9137282499159616) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[10],q[8];
rz(-pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-pi/2) q[8];
rz(2.8592890227932743) q[8];
sx q[8];
rz(-1.3790253791336597) q[9];
cx q[3],q[9];
rz(-pi/2) q[3];
cx q[13],q[3];
rz(pi/2) q[13];
rz(pi/2) q[3];
cx q[3],q[16];
rz(-pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-pi/2) q[16];
rz(pi/4) q[16];
cx q[16],q[0];
rz(-pi/4) q[0];
cx q[16],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[10],q[0];
rz(5.675829092454075) q[0];
cx q[10],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(2.0979758813938383) q[10];
cx q[10],q[0];
rz(-2.0979758813938383) q[0];
cx q[10],q[0];
rz(2.0979758813938383) q[0];
rz(3.814114822202054) q[0];
rz(2.948568405439827) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/4) q[10];
cx q[6],q[13];
cx q[13],q[6];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[14];
rz(0) q[14];
sx q[14];
rz(3.0559819488060564) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[13],q[14];
rz(pi/4) q[13];
cx q[13],q[12];
rz(-pi/4) q[12];
cx q[13],q[12];
rz(pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-3.0148788870914007) q[12];
rz(pi/2) q[12];
cx q[5],q[13];
rz(-0.7994756191659416) q[13];
cx q[5],q[13];
rz(0.7994756191659416) q[13];
cx q[13],q[15];
cx q[15],q[13];
cx q[13],q[15];
rz(3.6874500927582607) q[13];
rz(0.42422325120604304) q[13];
rz(0) q[15];
sx q[15];
rz(3.354257039733413) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[17],q[15];
rz(0) q[15];
sx q[15];
rz(2.928928267446173) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[17],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[5];
sx q[6];
rz(pi/4) q[6];
cx q[7],q[14];
cx q[14],q[7];
cx q[7],q[14];
rz(-pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[12];
rz(0) q[12];
sx q[12];
rz(3.051085412305836) q[12];
sx q[12];
rz(3*pi) q[12];
rz(0) q[14];
sx q[14];
rz(3.23209989487375) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[14],q[12];
rz(-pi/2) q[12];
rz(3.0148788870914007) q[12];
rz(-pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(1.3711899079970113) q[14];
cx q[14],q[11];
rz(-1.3711899079970113) q[11];
cx q[14],q[11];
rz(1.3711899079970113) q[11];
rz(pi/2) q[11];
rz(5.294640141447921) q[14];
cx q[4],q[12];
cx q[12],q[4];
cx q[4],q[12];
cx q[12],q[10];
rz(-pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-pi/2) q[10];
rz(-pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(2.752491027971619) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[5];
cx q[5],q[7];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[7];
cx q[7],q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[15],q[5];
rz(5.917995677103373) q[5];
cx q[15],q[5];
rz(pi/4) q[15];
rz(-pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
x q[7];
cx q[8],q[6];
cx q[6],q[8];
cx q[8],q[6];
rz(1.693054741546351) q[6];
rz(2.123722631871222) q[6];
rz(1.3790253791336597) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(1.1000979943488012) q[9];
cx q[1],q[9];
rz(-1.1000979943488012) q[9];
cx q[1],q[9];
id q[1];
cx q[3],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(1.2696321193044462) q[1];
cx q[16],q[1];
rz(-1.2696321193044462) q[1];
cx q[16],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
x q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(1.2736466418709744) q[16];
sx q[16];
rz(6.413643071095711) q[16];
sx q[16];
rz(12.714680150944462) q[16];
rz(-0.2886533168844736) q[16];
cx q[18],q[3];
rz(-2.896406417816389) q[3];
cx q[18],q[3];
rz(-pi/2) q[18];
rz(-pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(6.42384952966368) q[18];
sx q[18];
rz(5*pi/2) q[18];
rz(2.896406417816389) q[3];
rz(0) q[3];
sx q[3];
rz(5.693366014648527) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[6],q[16];
rz(-2.123722631871222) q[16];
sx q[16];
rz(0.9146817029875551) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[6],q[16];
rz(0) q[16];
sx q[16];
rz(5.368503604192031) q[16];
sx q[16];
rz(11.837153909525075) q[16];
rz(pi/4) q[16];
rz(-pi/2) q[16];
rz(0.379165691449177) q[16];
rz(pi/4) q[6];
cx q[6],q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[8],q[1];
rz(0.8775804604508068) q[1];
cx q[8],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[1],q[8];
rz(6.234522192808055) q[8];
cx q[1],q[8];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[2],q[6];
rz(pi/4) q[2];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(0) q[8];
sx q[8];
rz(5.145853832490797) q[8];
sx q[8];
rz(3*pi) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(4.88842240042282) q[9];
sx q[9];
rz(3.5858297347552193) q[9];
sx q[9];
rz(12.16775996783823) q[9];
rz(3.417575848859097) q[9];
sx q[9];
rz(2.4424727468962266) q[9];
rz(0.9018776765035191) q[9];
sx q[9];
rz(8.590462247344515) q[9];
sx q[9];
rz(15.530950753794063) q[9];
cx q[9],q[3];
rz(0) q[3];
sx q[3];
rz(0.5898192925310588) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[9],q[3];
rz(-0.7473713414163983) q[3];
cx q[0],q[3];
rz(-3.814114822202054) q[3];
sx q[3];
rz(0.29269425508419245) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[0],q[3];
sx q[0];
cx q[11],q[0];
rz(2.4063907875026804) q[0];
x q[11];
cx q[12],q[11];
rz(5.778060124750015) q[11];
cx q[11],q[16];
rz(3.19661128790756) q[12];
sx q[12];
rz(4.906338928531718) q[12];
sx q[12];
rz(12.839687419802473) q[12];
cx q[14],q[0];
rz(-2.4063907875026804) q[0];
cx q[14],q[0];
rz(pi/4) q[0];
cx q[0],q[18];
cx q[13],q[14];
rz(-5.163512160233441) q[13];
rz(pi/2) q[13];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/4) q[14];
cx q[12],q[14];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi/4) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-5.778060124750015) q[16];
sx q[16];
rz(1.4951547237451976) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[11],q[16];
rz(3.88004417933463) q[11];
rz(3.678233871470687) q[11];
rz(0) q[16];
sx q[16];
rz(4.788030583434389) q[16];
sx q[16];
rz(14.823672394070218) q[16];
rz(4.707182541249649) q[16];
sx q[16];
rz(4.896089868202717) q[16];
sx q[16];
rz(11.993231440266833) q[16];
rz(5.103297849036892) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-pi/4) q[18];
cx q[0],q[18];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/4) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[2],q[18];
rz(-pi/4) q[18];
cx q[2],q[18];
rz(pi/4) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
x q[18];
rz(-6.060915276166127) q[18];
rz(pi/2) q[18];
cx q[12],q[18];
rz(0) q[12];
sx q[12];
rz(6.214618080261006) q[12];
sx q[12];
rz(3*pi) q[12];
rz(0) q[18];
sx q[18];
rz(0.06856722691858064) q[18];
sx q[18];
rz(3*pi) q[18];
cx q[12],q[18];
rz(-pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(-pi/2) q[18];
rz(6.060915276166127) q[18];
rz(0) q[3];
sx q[3];
rz(5.990491052095393) q[3];
sx q[3];
rz(13.986264124387832) q[3];
rz(pi) q[3];
x q[3];
rz(pi/2) q[3];
cx q[17],q[3];
cx q[3],q[17];
cx q[17],q[1];
rz(5.800306186318321) q[1];
cx q[17],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(5.384146089338436) q[1];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[15],q[17];
rz(-pi/4) q[17];
cx q[15],q[17];
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
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[5],q[13];
rz(0) q[13];
sx q[13];
rz(1.0163993236739337) q[13];
sx q[13];
rz(3*pi) q[13];
rz(0) q[5];
sx q[5];
rz(5.2667859835056525) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[5],q[13];
rz(-pi/2) q[13];
rz(5.163512160233441) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/4) q[13];
rz(-pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/4) q[5];
cx q[5],q[17];
rz(-pi/4) q[17];
cx q[5],q[17];
rz(pi/4) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/4) q[5];
cx q[5],q[16];
rz(-pi/4) q[16];
cx q[5],q[16];
rz(pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[6],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
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
rz(pi/2) q[0];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[2],q[6];
rz(pi/4) q[2];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[2],q[6];
rz(-pi/4) q[6];
cx q[2],q[6];
rz(pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/4) q[6];
cx q[8],q[3];
cx q[3],q[8];
cx q[8],q[3];
rz(-pi/4) q[3];
rz(pi/4) q[3];
cx q[8],q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[3],q[8];
rz(-pi/4) q[8];
cx q[3],q[8];
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
sx q[9];
rz(5.581660573050594) q[9];
sx q[9];
rz(5*pi/2) q[9];
rz(6.047190275182942) q[9];
rz(pi/2) q[9];
cx q[4],q[9];
rz(0) q[4];
sx q[4];
rz(2.2088409879519086) q[4];
sx q[4];
rz(3*pi) q[4];
rz(0) q[9];
sx q[9];
rz(2.2088409879519086) q[9];
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
rz(2.714473984134606) q[4];
cx q[7],q[4];
rz(-2.714473984134606) q[4];
cx q[7],q[4];
rz(3.1857402673520303) q[4];
rz(-0.5973172700745359) q[4];
cx q[11],q[4];
rz(-3.678233871470687) q[4];
sx q[4];
rz(0.043734773533070204) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[11],q[4];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[14],q[11];
rz(3.9430282586635013) q[11];
cx q[14],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-1.8659396832971993) q[11];
rz(pi/2) q[11];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[13];
rz(-pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-pi/2) q[13];
rz(0) q[4];
sx q[4];
rz(6.239450533646516) q[4];
sx q[4];
rz(13.700329102314601) q[4];
rz(-pi/2) q[4];
cx q[15],q[4];
cx q[15],q[2];
cx q[2],q[15];
cx q[15],q[2];
rz(pi/2) q[4];
cx q[17],q[4];
rz(4.637299802434674) q[4];
cx q[17],q[4];
rz(-pi/2) q[9];
rz(-6.047190275182942) q[9];
cx q[10],q[9];
cx q[9],q[10];
rz(3.5274582478347285) q[10];
sx q[10];
rz(6.616584475908057) q[10];
sx q[10];
rz(10.295289183302026) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[9],q[7];
cx q[7],q[9];
rz(2.0923355874294405) q[7];
sx q[7];
rz(5.628287136313856) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[10];
cx q[10],q[7];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[11];
rz(0) q[10];
sx q[10];
rz(4.260919418932052) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[11];
sx q[11];
rz(2.022265888247534) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[10],q[11];
rz(-pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(-pi/2) q[11];
rz(1.8659396832971993) q[11];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[6],q[7];
rz(-pi/4) q[7];
cx q[6],q[7];
rz(pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
id q[9];
sx q[9];
cx q[0],q[9];
x q[0];
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
