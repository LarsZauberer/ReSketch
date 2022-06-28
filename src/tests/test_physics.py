from src.agent_modules.physics import Physic_Engine


class Test_Engine:
    eng = Physic_Engine(mass=0.1, friction=0.1)
    
    def test_acceleration_1(self):
        a = self.eng.calc_acceleration([0, 0])
        assert a == [0, 0]
    
    def test_acceleration_2(self):
        a = self.eng.calc_acceleration([10, -10])
        assert float("{:.3f}".format(a[0])) == 100
        assert float("{:.3f}".format(a[1])) == -100
    
    def test_friction_1(self):
        f = self.eng.calc_friction()
        assert f == [0, 0]
        
    def test_friction_2(self):
        eng = Physic_Engine(mass=0.1, friction=0.1)
        eng.velocity = [1, 1]
        f = eng.calc_friction()
        assert float("{:.3f}".format(f[0])) == 0.098
        assert float("{:.3f}".format(f[1])) == 0.098
        
    def test_friction_3(self):
        eng = Physic_Engine(mass=0.1, friction=0.1)
        eng.velocity = [1, 0]
        f = eng.calc_friction()
        assert float("{:.3f}".format(f[0])) == 0.098
        assert float("{:.3f}".format(f[1])) == 0.0
    
    def test_friction_4(self):
        eng = Physic_Engine(mass=0.1, friction=0.1)
        eng.velocity = [0, 1]
        f = eng.calc_friction()
        assert float("{:.3f}".format(f[0])) == 0.0
        assert float("{:.3f}".format(f[1])) == 0.098
    
    def test_pos(self):
        eng = Physic_Engine(mass=0.1, friction=0.1)
        pos = [0, 0]
        pos = eng.calc_position_step(pos, [1, 1])
        assert pos == [10, 10]
        assert eng.velocity == [9.9019, 9.9019]
        pos = eng.calc_position_step(pos, [0, 0])
        assert pos == [20, 20]
        assert float("{:.3f}".format(eng.velocity[0])) == 9.804
        assert float("{:.3f}".format(eng.velocity[1])) == 9.804
        