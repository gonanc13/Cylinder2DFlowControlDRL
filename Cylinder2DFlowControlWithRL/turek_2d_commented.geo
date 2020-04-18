// Position of jets in radians wrt flow direction (pi/2, 3pi/2)
jet_positions[] = {1.5707963267948966, 4.71238898038469};

// Define parameters. These can then be quickly modified in the GUI.
DefineConstant[
length = {2.2, Name "Channel length"}
front_distance = {0.2, Name "Cylinder center distance to inlet"}
bottom_distance = {0.2, Name "Cylinder center distance from bottom"}
jet_radius = {0.05, Name "Cylinder radius"}
jet_width = {10*Pi/180, Name "Jet width in radians"}
width = {0.41, Name "Channel width"}
cylinder_size = {0.02, Name "Mesh size on cylinder"}
box_size = {0.05, Name "Mesh size on wall"}
coarse_size = {0.1, Name "Mesh size close to the outflow"}
coarse_distance = {0.5, Name "Distance (x) from the cylinder's right-most point where coarsening starts"}
];

// Seed the cylinder center's identifier and create the center point
center = newp;
Point(center) = {0, 0, 0, cylinder_size};

// Obtain number of jets from no. elements in jet_positions array
n = #jet_positions[];

// Cylinder radius
radius = jet_radius;

// Start definition of cylinder surface (curves). Note: it is defined in CCW sense

If(n > 0)
  // Create empty lists for ...
  cylinder[] = {}; // List of curves (surfaces) of the cylinder
  lower_bound[] = {}; // List of boundary points of jet surfaces that are CW to arch centers
  uppper_bound[] = {}; // List of boundary points of jet surfaces that are CCW to arch centers

  //  Define jet surfaces for each jet
  For i In {0:(n-1)}

      // Angle wrt incoming flow
      angle = jet_positions[i];

      // Define first point of jet surface (CW to jet arch center)
      x = radius*Cos(angle-jet_width/2);
      y = radius*Sin(angle-jet_width/2);
      p = newp;
      Point(p) = {x, y, 0, cylinder_size};
      lower_bound[] += {p};

      // Define second point of jet surface (arch center)
      x0 = radius*Cos(angle);
      y0 = radius*Sin(angle);
      arch_center = newp;
      Point(arch_center) = {x0, y0, 0, cylinder_size};

      // Define third point of jet surface (CCW to jet arch center)
      x = radius*Cos(angle+jet_width/2);
      y = radius*Sin(angle+jet_width/2);
      q = newp;
      Point(q) = {x, y, 0, cylinder_size};
      upper_bound[] += {q};
  
      // Draw the piece; p to angle (Draw half the arch, from CW boundary point (p) to arch center)
      l = newl;
      Circle(l) = {p, center, arch_center}; 
      // Let each yet be marked as a different surface
      Physical Line(5+i) = {l}; // Add half arch to this jet's physical surface
      cylinder[] += {l}; // Add half arch to cylinder list

      // Draw the piece; angle to q (Draw half the arch, from arch center to CCW boundary point (q))
      l = newl;
      Circle(l) = {arch_center, center, q}; 
      // Let each yet be marked as a different surface
      Physical Line(5+i) += {l}; // Add half arch to this jet's physical surface
      cylinder[] += {l}; // Add half arch to cylinder list

      // Each jet surface (arch from p to q) is defined as a physical line
  EndFor

  // Fill in the rest of the cylinder surfaces. These are no slip surfaces (and are defined in CCW sense)
  lower_bound[] += {lower_bound[0]}; // Duplicate first point of lower bounds, to adapt to the way curves are defined
  Physical Line(4) = {};  // No slip cylinder surfaces
  For i In {0:(n-1)}
    p = upper_bound[i]; // Start point of each cylinder surface outside of jets
    q = lower_bound[i+1]; // End point of each cylinder surface outside of jets

    pc[] = Point{p}; // Get coordinates in list format
    qc[] = Point{q}; // Get coordinates

    // Compute the angle (wrt incoming flow)
    angle_p = Atan2(pc[1], pc[0]); // Arc tangent of the first expression divided by the second
    angle_p = (angle_p > 0) ? angle_p : (2*Pi + angle_p); // If angle is negative, substract abs from 2pi

    angle_q = Atan2(qc[1], qc[0]);
    angle_q = (angle_q > 0) ? angle_q : (2*Pi + angle_q);

    // Find angle difference p to q. Note this one always gives right angle
    angle = angle_q - angle_p; // front back
    angle = (angle < 0) ? angle + 2*Pi : angle; // check also back front
    Printf("%g", angle); // Print this angle difference

    // If arc from p to q greater than Pi, then we need to insert point (maximum arc angle in Gmsh is pi)
    If(angle > Pi)
      // Create point halfway from p to q
      // Rotate by half angle(p,q) a duplicate of p CCW wrt an axis parallel to z passing through center
      half[] = Rotate {{0, 0, 1}, {0, 0, 0}, angle/2} {Duplicata{Point{p};}};         

      l = newl;
      Circle(l) = {p, center, half}; 
      // Let each yet be marked as a different surface
      Physical Line(4) += {l}; // Add to no slip cylinder surfaces
      cylinder[] += {l}; // Add to cylinder lines list. Note it now contains first jet surfaces then no slip surfaces.

      l = newl;
      Circle(l) = {half, center, q}; 
      // Let each yet be marked as a different surface
      Physical Line(4) += {l};
      cylinder[] += {l};                     
    Else
      l = newl;
      Circle(l) = {p, center, q}; 
      // Let each yet be marked as a different surface
      Physical Line(4) += {l};
      cylinder[] += {l};
    EndIf
  EndFor
// Just the circle (if number of jet surfaces = 0)
Else
   p = newp; 
   Point(p) = {-jet_radius, 0, 0, cylinder_size};
   Point(p+1) = {0, jet_radius, 0, cylinder_size};
   Point(p+2) = {jet_radius, 0, 0, cylinder_size};
   Point(p+3) = {0, -jet_radius, 0, cylinder_size};
	
   l = newl;
   Circle(l) = {p, center, p+1};
   Circle(l+1) = {p+1, center, p+2};
   Circle(l+2) = {p+2, center, p+3};
   Circle(l+3) = {p+3, center, p};

   cylinder[] = {l, l+1, l+2, l+3};	// Defined clockwise
   Physical Line(4) = {cylinder[]};
EndIf

// Create the channel (Domain exterior boundary)
p = newp;
Point(p) = {-front_distance, -bottom_distance, 0, box_size}; // bottom-left corner
Point(p+1) = {jet_radius+coarse_distance, -bottom_distance, 0, coarse_size}; // bottom start of coarse zone
Point(p+2) = {-front_distance+length, -bottom_distance, 0, coarse_size}; // bottom-right corner
Point(p+5) = {-front_distance, -bottom_distance+width, 0, box_size}; // top-left corner
Point(p+4) = {jet_radius+coarse_distance, -bottom_distance+width, 0, coarse_size}; // top start of coarse zone
Point(p+3) = {-front_distance+length, -bottom_distance+width, 0, coarse_size}; // top-right corner

l = newl;
// A no slip wall (bottom wall)
Line(l) = {p, p+1};
Line(l+1) = {p+1, p+2};
Physical Line(1) = {l, l+1};

// Outflow (right wall)
Line(l+2) = {p+2, p+3};
Physical Line(2) = {l+2};

// Top no slip wall (top wall)
Line(l+3) = {p+3, p+4};
Line(l+4) = {p+4, p+5};
Physical Line(1) += {l+3, l+4};

// Inlet
Line(l+5) = {p+5, p};
Physical Line(3) = {l+5};

// Coarse line
Line(l+6) = {p+1, p+4};

// Create line loop for coarse area
coarse = newll;
Line Loop(coarse) = {(l+1), (l+2), (l+3), -(l+6)};

// Create surface and physical surface for coarse area
s = news;
Plane Surface(s) = {coarse};
Physical Surface(1) = {s};

// Create list of lines that contain the cylinder (== fine mesh zone)
cframe[] = {l, (l+6), l+4, l+5};

// The surface to be mesh;
outer = newll;
Line Loop(outer) = {cframe[]}; // Outer line loop

inner = newll;
Line Loop(inner) = {cylinder[]}; // Inner line loop (cylinder)

s = news;
Plane Surface(s) = {inner, outer}; // Should be outer, inner, no??
Physical Surface(1) += {s}; // Physical surface 1 contains whole domain with a whole of the cylinder

// First the jet and no slip surfaces of the cylinder are defined. Each jet surface is a physical line and all the no slip
// cylinder surfaces are another. Then the domain is created.

// Characteristic Length{cylinder[]} = cylinder_size;
// Characteristic Length{coarse[]} = coarse_size;
// Characteristic Length{cframe[]} = box_size;