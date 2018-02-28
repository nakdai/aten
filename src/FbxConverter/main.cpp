#include "aten.h"
#include "atenscene.h"

#include "FbxImporter.h"
#include "MdlExporter.h"
#include "MtrlExporter.h"

#include <cmdline.h>
#include <imgui.h>

static const int WIDTH = 1280;
static const int HEIGHT = 720;

static const char* TITLE = "FbxConverter";

struct Options {
	std::string input;
	std::string output;

	std::string inputBasepath;
	std::string inputFilename;
} g_opt;

aten::deformable g_mdl;

static aten::PinholeCamera g_camera;
static bool g_isCameraDirty = false;

static uint32_t g_lodTriCnt = 0;
static bool g_isWireFrame = true;
static bool g_isUpdateBuffer = false;
static bool g_displayLOD = true;

static bool g_willShowGUI = true;

static bool g_isMouseLBtnDown = false;
static bool g_isMouseRBtnDown = false;
static int g_prevX = 0;
static int g_prevY = 0;

void onRun()
{
	if (g_isCameraDirty) {
		g_camera.update();

		auto camparam = g_camera.param();
		camparam.znear = real(0.1);
		camparam.zfar = real(10000.0);

		g_isCameraDirty = false;
	}

	g_mdl.update(aten::mat4());
	aten::DeformableRenderer::render(&g_camera, &g_mdl);
}

void onClose()
{

}

void onMouseBtn(bool left, bool press, int x, int y)
{
	g_isMouseLBtnDown = false;
	g_isMouseRBtnDown = false;

	if (press) {
		g_prevX = x;
		g_prevY = y;

		g_isMouseLBtnDown = left;
		g_isMouseRBtnDown = !left;
	}
}

void onMouseMove(int x, int y)
{
	if (g_isMouseLBtnDown) {
		aten::CameraOperator::rotate(
			g_camera,
			WIDTH, HEIGHT,
			g_prevX, g_prevY,
			x, y);
		g_isCameraDirty = true;
	}
	else if (g_isMouseRBtnDown) {
		aten::CameraOperator::move(
			g_camera,
			g_prevX, g_prevY,
			x, y,
			real(0.001));
		g_isCameraDirty = true;
	}

	g_prevX = x;
	g_prevY = y;
}

void onMouseWheel(int delta)
{
	aten::CameraOperator::dolly(g_camera, delta * real(0.1));
	g_isCameraDirty = true;
}

void onKey(bool press, aten::Key key)
{
	static const real offset = real(5);

	if (press) {
		if (key == aten::Key::Key_F1) {
			g_willShowGUI = !g_willShowGUI;
			return;
		}
	}

	if (press) {
		switch (key) {
		case aten::Key::Key_W:
		case aten::Key::Key_UP:
			aten::CameraOperator::moveForward(g_camera, offset);
			break;
		case aten::Key::Key_S:
		case aten::Key::Key_DOWN:
			aten::CameraOperator::moveForward(g_camera, -offset);
			break;
		case aten::Key::Key_D:
		case aten::Key::Key_RIGHT:
			aten::CameraOperator::moveRight(g_camera, offset);
			break;
		case aten::Key::Key_A:
		case aten::Key::Key_LEFT:
			aten::CameraOperator::moveRight(g_camera, -offset);
			break;
		case aten::Key::Key_Z:
			aten::CameraOperator::moveUp(g_camera, offset);
			break;
		case aten::Key::Key_X:
			aten::CameraOperator::moveUp(g_camera, -offset);
			break;
		default:
			break;
		}

		g_isCameraDirty = true;
	}
}

bool parseOption(
	int argc, char* argv[],
	cmdline::parser& cmd,
	Options& opt)
{
	{
		cmd.add<std::string>("input", 'i', "input filename", true);
		cmd.add<std::string>("output", 'o', "output filename base", false, "result");

		cmd.add<std::string>("help", '?', "print usage", false);
	}

	bool isCmdOk = cmd.parse(argc, argv);

	if (cmd.exist("help")) {
		std::cerr << cmd.usage();
		return false;
	}

	if (!isCmdOk) {
		std::cerr << cmd.error() << std::endl << cmd.usage();
		return false;
	}

	if (cmd.exist("input")) {
		opt.input = cmd.get<std::string>("input");
	}
	else {
		std::cerr << cmd.error() << std::endl << cmd.usage();
		return false;
	}

	if (cmd.exist("output")) {
		opt.output = cmd.get<std::string>("output");
	}
	else {
		// TODO
		opt.output = "result.sbvh";
	}

	return true;
}

int main(int argc, char* argv[])
{
	aten::window::SetCurrentDirectoryFromExe();

	aten::FbxImporter importer;

#if 1
	importer.setIgnoreTexIdx(0);
	importer.open("../../asset/unitychan/unitychan.fbx");
	MdlExporter::exportMdl(48, "unitychan.mdl", &importer);
	MtrlExporter::exportMaterial("unitychan_mtrl.xml", &importer);
#endif

	aten::window::init(
		WIDTH, HEIGHT,
		TITLE,
		onClose,
		onMouseBtn,
		onMouseMove,
		onMouseWheel,
		onKey);

	aten::DeformableRenderer::init(
		WIDTH, HEIGHT,
		"../shader/skinning_vs.glsl",
		"../shader/skinning_fs.glsl");

	g_mdl.read("unitychan.mdl");

	aten::ImageLoader::setBasePath("../../asset/unitychan/Texture");
	aten::MaterialLoader::load("unitychan_mtrl.xml");

	auto textures = aten::texture::getTextures();
	for (auto tex : textures) {
		tex->initAsGLTexture();
	}

	// TODO
	aten::vec3 pos(0, 1, 10);
	aten::vec3 at(0, 1, 1);
	real vfov = real(45);

	g_camera.init(
		pos,
		at,
		aten::vec3(0, 1, 0),
		vfov,
		WIDTH, HEIGHT);

	aten::window::run(onRun);

	aten::window::terminate();

	return 1;
}
