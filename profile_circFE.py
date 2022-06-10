import Circ_FE
import cProfile

profiled = cProfile.run("Circ_FE.main(10)")
print(profiled)
