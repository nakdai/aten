#include "scenedefs.h"
#include "atenscene.h"

void CornellBoxScene::makeScene(aten::scene* scene)
{
	auto emit = new aten::emissive(aten::vec3(36, 36, 36));
	//auto emit = new aten::emissive(aten::vec3(3, 3, 3));

	auto light = new aten::sphere(
		aten::vec3(50.0, 75.0, 81.6),
		5.0,
		emit);

	double r = 1e3;

	auto left = new aten::sphere(
		aten::vec3(r + 1, 40.8, 81.6),
		r,
		new aten::lambert(aten::vec3(0.75f, 0.25f, 0.25f)));

	auto right = new aten::sphere(
		aten::vec3(-r + 99, 40.8, 81.6),
		r,
		new aten::lambert(aten::vec3(0.25, 0.25, 0.75)));

	auto wall = new aten::sphere(
		aten::vec3(50, 40.8, r),
		r,
		new aten::lambert(aten::vec3(0.75, 0.75, 0.75)));

	auto floor = new aten::sphere(
		aten::vec3(50, r, 81.6),
		r,
		new aten::lambert(aten::vec3(0.75, 0.75, 0.75)));

	auto ceil = new aten::sphere(
		aten::vec3(50, -r + 81.6, 81.6),
		r,
		new aten::lambert(aten::vec3(0.75, 0.75, 0.75)));

	// �΋�.
	auto green = new aten::sphere(
		aten::vec3(65, 20, 20),
		20,
		new aten::lambert(aten::vec3(0.25, 0.75, 0.25)));
	//new aten::lambert(aten::vec3(1, 1, 1), tex));

	// ��.
	auto mirror = new aten::sphere(
		aten::vec3(27, 16.5, 47),
		16.5,
		new aten::specular(aten::vec3(0.99, 0.99, 0.99)));

#if 0
	// �K���X.
	auto glass = new aten::sphere(
		aten::vec3(77, 16.5, 78),
		16.5,
		new aten::refraction(aten::vec3(0.99, 0.99, 0.99), 1.5));
#else
	aten::AssetManager::registerMtrl(
		"Material.001",
		new aten::lambert(aten::vec3(0.2, 0.2, 0.7)));

	auto obj = aten::ObjLoader::load("../../asset/suzanne/suzanne.obj");

	aten::mat4 mtxL2W;
	mtxL2W.asRotateByY(Deg2Rad(-25));

	aten::mat4 mtxT;
	mtxT.asTrans(aten::vec3(77, 16.5, 78));

	aten::mat4 mtxS;
	mtxS.asScale(10);

	mtxL2W = mtxT * mtxL2W * mtxS;

	auto glass = new aten::instance<aten::object>(obj, mtxL2W);
#endif

	aten::Light* l = new aten::AreaLight(light, emit->color());

	scene->addLight(l);

#if 1
	scene->add(light);
	scene->add(left);
	scene->add(right);
	scene->add(wall);
	scene->add(floor);
	scene->add(ceil);
	scene->add(green);
	scene->add(mirror);
	scene->add(glass);
#else
	scene->add(light);
	scene->add(glass);
#endif
}

void CornellBoxScene::getCameraPosAndAt(
	aten::vec3& pos,
	aten::vec3& at,
	real& fov)
{
	pos = aten::vec3(50.0, 52.0, 295.6);
	at = aten::vec3(50.0, 40.8, 119.0);
	fov = 30;
}

/////////////////////////////////////////////////////

void MtrlTestScene::makeScene(aten::scene* scene)
{
	auto s_blinn = new aten::sphere(aten::vec3(-1, 0, 0), 1.0, new aten::MicrofacetBlinn(aten::vec3(0.7, 0.6, 0.5), 200, 0.2));
	scene->add(s_blinn);

	auto s_ggx = new aten::sphere(aten::vec3(-3, 0, 0), 1.0, new aten::MicrofacetGGX(aten::vec3(0.7, 0.6, 0.5), 0.2, 0.2));
	scene->add(s_ggx);

	auto s_beckman = new aten::sphere(aten::vec3(+1, 0, 0), 1.0, new aten::MicrofacetBeckman(aten::vec3(0.7, 0.6, 0.5), 0.2, 0.2));
	scene->add(s_beckman);

	auto s_glass = new aten::sphere(aten::vec3(+3, 0, 0), 1.0, new aten::specular(aten::vec3(0.7, 0.6, 0.5)));
	scene->add(s_glass);
}

void MtrlTestScene::getCameraPosAndAt(
	aten::vec3& pos,
	aten::vec3& at,
	real& fov)
{
	pos = aten::vec3(0, 0, 13);
	at = aten::vec3(0, 0, 0);
	fov = 30;
}

/////////////////////////////////////////////////////

void PointLightScene::makeScene(aten::scene* scene)
{
	auto emit = new aten::emissive(aten::vec3(36.0, 36.0, 36.0));

	auto light = new aten::sphere(
		aten::vec3(50.0, 90.0, 81.6),
		15.0,
		emit);

	double r = 1e5;

	auto floor = new aten::sphere(
		aten::vec3(0, -r, 0),
		r,
		new aten::lambert(aten::vec3(0.75, 0.75, 0.75)));

	// �΋�.
	auto green = new aten::sphere(
		aten::vec3(65, 20, 20),
		20,
		new aten::lambert(aten::vec3(0.25, 0.75, 0.25)));

	//scene->add(light);
	scene->add(floor);
	scene->add(green);

	aten::Light* l = new aten::PointLight(aten::vec3(50.0, 90.0, 81.6), aten::vec3(36.0, 36.0, 36.0), 0, 0.1, 0);
	//aten::Light* l = new aten::AreaLight(light, emit->color());

	scene->addLight(l);
}

void PointLightScene::getCameraPosAndAt(
	aten::vec3& pos,
	aten::vec3& at,
	real& fov)
{
	pos = aten::vec3(50.0, 52.0, 295.6);
	at = aten::vec3(50.0, 40.8, 119.0);
	fov = 30;
}

/////////////////////////////////////////////////////

void SpotLightScene::makeScene(aten::scene* scene)
{
	double r = 1e5;

	auto floor = new aten::sphere(
		aten::vec3(0, -r, 0),
		r,
		new aten::lambert(aten::vec3(0.75, 0.75, 0.75)));

	// �΋�.
	auto green = new aten::sphere(
		aten::vec3(65, 20, 20),
		20,
		new aten::lambert(aten::vec3(0.25, 0.75, 0.25)));

	auto red = new aten::sphere(
		aten::vec3(25, 20, 20),
		20,
		new aten::lambert(aten::vec3(0.75, 0.25, 0.25)));

	//scene->add(light);
	scene->add(floor);
	scene->add(green);
	scene->add(red);

	aten::Light* l = new aten::SpotLight(
		aten::vec3(65, 90, 20),
		aten::vec3(0, -1, 0),
		aten::vec3(36.0, 36.0, 36.0),
		0, 0.1, 0,
		Deg2Rad(30),
		Deg2Rad(60),
		1);
	//aten::Light* l = new aten::AreaLight(light, emit->color());

	scene->addLight(l);
}

void SpotLightScene::getCameraPosAndAt(
	aten::vec3& pos,
	aten::vec3& at,
	real& fov)
{
	pos = aten::vec3(50.0, 52.0, 295.6);
	at = aten::vec3(50.0, 40.8, 119.0);
	fov = 30;
}

/////////////////////////////////////////////////////

void DirectionalLightScene::makeScene(aten::scene* scene)
{
	auto emit = new aten::emissive(aten::vec3(36.0, 36.0, 36.0));

	auto light = new aten::sphere(
		aten::vec3(50.0, 90.0, 81.6),
		15.0,
		emit);

	double r = 1e5;

	auto floor = new aten::sphere(
		aten::vec3(0, -r, 0),
		r,
		new aten::lambert(aten::vec3(0.75, 0.75, 0.75)));

	// �΋�.
	auto green = new aten::sphere(
		aten::vec3(65, 20, 20),
		20,
		new aten::lambert(aten::vec3(0.25, 0.75, 0.25)));

	//scene->add(light);
	scene->add(floor);
	scene->add(green);

	aten::Light* l = new aten::DirectionalLight(aten::vec3(1, -1, 1), aten::vec3(36.0, 36.0, 36.0));
	//aten::Light* l = new aten::AreaLight(light, emit->color());

	scene->addLight(l);
}

void DirectionalLightScene::getCameraPosAndAt(
	aten::vec3& pos,
	aten::vec3& at,
	real& fov)
{
	pos = aten::vec3(50.0, 52.0, 295.6);
	at = aten::vec3(50.0, 40.8, 119.0);
	fov = 30;
}

/////////////////////////////////////////////////////

void TexturesScene::makeScene(aten::scene* scene)
{
	auto albedo = aten::ImageLoader::load("../../asset/pbr_textures/Brick_baked/T_Brick_Baked_D.tga");
	auto nml = aten::ImageLoader::load("../../asset/pbr_textures/Brick_baked/T_Brick_Baked_N.tga");
	auto rough = aten::ImageLoader::load("../../asset/pbr_textures/Brick_baked/T_Brick_Baked_R.tga");
	auto nml_2 = aten::ImageLoader::load("../../asset/normalmap.png");
	aten::vec3 clr = aten::vec3(1, 1, 1);

	auto s_blinn = new aten::sphere(aten::vec3(-3, 0, 0), 1.0, new aten::MicrofacetBlinn(clr, 200, 0.2, albedo, nml));
	scene->add(s_blinn);
#if 1
	auto s_ggx = new aten::sphere(aten::vec3(-1, 0, 0), 1.0, new aten::MicrofacetGGX(clr, 0.2, 0.2, albedo, nml, rough));
	scene->add(s_ggx);

	auto s_beckman = new aten::sphere(aten::vec3(+1, 0, 0), 1.0, new aten::MicrofacetBeckman(clr, 0.2, 0.2, albedo, nml, rough));
	scene->add(s_beckman);

	auto s_lambert = new aten::sphere(aten::vec3(+3, 0, 0), 1.0, new aten::lambert(clr, albedo, nml));
	scene->add(s_lambert);

	auto s_spec = new aten::sphere(aten::vec3(-3, +2, 0), 1.0, new aten::specular(clr, nullptr, nml_2));
	scene->add(s_spec);

	auto s_ref = new aten::sphere(aten::vec3(-1, +2, 0), 1.0, new aten::specular(clr, nullptr, nml_2));
	scene->add(s_ref);
#endif
}

void TexturesScene::getCameraPosAndAt(
	aten::vec3& pos,
	aten::vec3& at,
	real& fov)
{
	pos = aten::vec3(0, 0, 13);
	at = aten::vec3(0, 0, 0);
	fov = 30;
}

/////////////////////////////////////////////////////

void DisneyMaterialTestScene::makeScene(aten::scene* scene)
{
	{
		aten::MaterialParameter param;
		param.baseColor = aten::vec3(0.82, 0.67, 0.16);
		param.roughness = 0.3;
		param.specular = 0.5;
		param.metallic = 0.5;

		auto m = new aten::DisneyBRDF(param);
		auto s = new aten::sphere(aten::vec3(0, 0, 0), 1.0, m);
		scene->add(s);
	}

	{
		auto m = new aten::lambert(aten::vec3(0.82, 0.67, 0.16));
		auto s = new aten::sphere(aten::vec3(-3, 0, 0), 1.0, m);
		scene->add(s);
	}

	aten::Light* dir = new aten::DirectionalLight(aten::vec3(-1, -1, -1), aten::vec3(0.5, 0.5, 0.5));
	//scene->addLight(dir);

#if 0
	{
		aten::DisneyBRDF::Parameter param;
		param.sheen = 0.5;

		auto m = new aten::DisneyBRDF(param);
		auto s = new aten::sphere(aten::vec3(-1, 0, 0), 1.0, m);
		scene->add(s);
	}

	{
		aten::DisneyBRDF::Parameter param;
		param.anisotropic = 0.5;

		auto m = new aten::DisneyBRDF(param);
		auto s = new aten::sphere(aten::vec3(+1, 0, 0), 1.0, m);
		scene->add(s);
	}

	{
		aten::DisneyBRDF::Parameter param;
		param.subsurface = 0.5;

		auto m = new aten::DisneyBRDF(param);
		auto s = new aten::sphere(aten::vec3(+3, 0, 0), 1.0, m);
		scene->add(s);
	}
#endif
}

void DisneyMaterialTestScene::getCameraPosAndAt(
	aten::vec3& pos,
	aten::vec3& at,
	real& fov)
{
	pos = aten::vec3(0, 0, 13);
	at = aten::vec3(0, 0, 0);
	fov = 30;
}

/////////////////////////////////////////////////////

void LayeredMaterialTestScene::makeScene(aten::scene* scene)
{
	auto spec = new aten::MicrofacetBlinn(aten::vec3(1, 1, 1), 200, 0.8);
	auto diff = new aten::lambert(aten::vec3(0.7, 0.0, 0.0));

	auto layer = new aten::LayeredBSDF();
	layer->add(spec);
	layer->add(diff);

	auto s_layer = new aten::sphere(aten::vec3(-3, 0, 0), 1.0, layer);
	scene->add(s_layer);

	auto s_diff = new aten::sphere(aten::vec3(-1, 0, 0), 1.0, diff);
	scene->add(s_diff);

	auto s_spec = new aten::sphere(aten::vec3(+1, 0, 0), 1.0, new aten::MicrofacetBlinn(aten::vec3(0.7, 0, 0), 200, 0.8));
	scene->add(s_spec);
}

void LayeredMaterialTestScene::getCameraPosAndAt(
	aten::vec3& pos,
	aten::vec3& at,
	real& fov)
{
	pos = aten::vec3(0, 0, 13);
	at = aten::vec3(0, 0, 0);
	fov = 30;
}

/////////////////////////////////////////////////////

void ObjCornellBoxScene::makeScene(aten::scene* scene)
{
	aten::AssetManager::registerMtrl(
		"backWall",
		new aten::lambert(aten::vec3(0.580000, 0.568000, 0.544000)));
	aten::AssetManager::registerMtrl(
		"ceiling",
		new aten::lambert(aten::vec3(0.580000, 0.568000, 0.544000)));
	aten::AssetManager::registerMtrl(
		"floor",
		new aten::lambert(aten::vec3(0.580000, 0.568000, 0.544000)));
	aten::AssetManager::registerMtrl(
		"leftWall",
		new aten::lambert(aten::vec3(0.504000, 0.052000, 0.040000)));

	auto emit = new aten::emissive(aten::vec3(36, 33, 24));
	aten::AssetManager::registerMtrl(
		"light",
		emit);

	aten::AssetManager::registerMtrl(
		"rightWall",
		new aten::lambert(aten::vec3(0.112000, 0.360000, 0.072800)));
	aten::AssetManager::registerMtrl(
		"shortBox",
		new aten::lambert(aten::vec3(0.580000, 0.568000, 0.544000)));
	aten::AssetManager::registerMtrl(
		"tallBox",
		new aten::lambert(aten::vec3(0.580000, 0.568000, 0.544000)));

	std::vector<aten::object*> objs;

#if 1
	aten::ObjLoader::load(objs, "../../asset/cornellbox/orig.obj");

	auto light = new aten::instance<aten::object>(objs[0], aten::mat4::Identity);
	auto box = new aten::instance<aten::object>(objs[1], aten::mat4::Identity);

	scene->add(light);
	scene->add(box);

	auto areaLight = new aten::AreaLight(light, emit->param().baseColor);
	scene->addLight(areaLight);
#else
	aten::ObjLoader::load(objs, "../../asset/cornellbox/orig_nolight.obj");
	auto box = new aten::instance<aten::object>(objs[0], aten::mat4::Identity);
	scene->add(box);
#endif
}

void ObjCornellBoxScene::getCameraPosAndAt(
	aten::vec3& pos,
	aten::vec3& at,
	real& fov)
{
	pos = aten::vec3(0.f, 1.f, 3.f);
	at = aten::vec3(0.f, 1.f, 0.f);
	fov = 45;
}