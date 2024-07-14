require 'nn'
require 'paths'
require 'cunn'
require 'hdf5'

cmd = torch.CmdLine()
cmd:option('-vid' , '')
cmd:option('-crop' , '')
cmd:option('-feat' , '')
params = cmd:parse(arg)
torch.setdefaulttensortype('torch.FloatTensor')
--load net
net = torch.load('vgg_face/VGG_FACE.t7')
net:evaluate()
net:cuda()

function load_faces(mv , vid)
	local fn = string.format('%s/%s/%s.h5' , params.crop , mv , vid)
	local file = hdf5.open(fn , 'r')
	local faces = file:read('faces'):all()
	local idx = file:read('idx'):all()
	if faces:dim() ~= 0 then
		faces = faces:permute(1,4,2,3)
	end
	--faces = faces:resize(idx:sum() , 3, 224 ,224)
	print (#faces)
	print (#idx)
	file:close()
	return faces , idx
end

mean = {129.1863,104.7624,93.5940}
function extract_feat(mv , vid)
	local faces, idx = load_faces(mv , vid)
	if faces:dim() == 0 then return nil end
	local trknum = idx:size()[1]
	local feat1 = torch.Tensor(idx:sum() , 4096):cuda()
	local feat2 = torch.Tensor(idx:sum() , 4096):cuda()
	local cnt = 0
    idx = idx:floor()
	for i=1,3 do faces[{{},i,{},{}}]:add(-mean[i]) end
	for i = 1,trknum do
		face = torch.Tensor(idx[i],3,224,224):copy(faces[{{cnt+1,cnt+idx[i]}}])
		prob = net(face:cuda())
		feat1[{{cnt+1,cnt+idx[i]}}] = net:get(33).output
		feat2[{{cnt+1,cnt+idx[i]}}] = net:get(36).output
		cnt = cnt+idx[i]
		print(i)
	end
	local dir = string.format('%s/%s' , params.feat , mv)
	local fn = string.format('%s/%s.h5' , dir , vid)
	if paths.dir(dir) == nil then paths.mkdir(dir) end
	local file = hdf5.open(fn , 'w')
	file:write('fc6' , feat1:double())
	file:write('fc7' , feat2:double())
	file:write('idx' , idx)
	file:close()
	print(mv)
end

--load mv
vidpath = params.vid
s = vidpath:split('/')
vid = s[#s]
mv = s[#s-1]
print (mv,vid)
extract_feat(mv , vid)
