include "mkl_vsl.f90" !necessary for intel mkl random numbers

module spin_systems
    contains
    subroutine binomialCoefficient(n, k, C)
        integer, intent(in)  :: n, k
        integer(8), intent(out) :: C
        !use, intrinsic :: iso_fortran_env, only: real64, int64
        !integer(int64) :: x
        integer :: i
        integer :: C_temp

        C_temp = 1

        if (k < 0 .or. k > n) then
        C_temp = 0
        else
        do i = 1, min(k, n-k)
            C_temp = C_temp * (n - i + 1) / i
            if (C_temp.GT.2147483647) then
                write(*,*) 'possible integer 32bit 2^32 = 2147483647 overflow'
                error stop
            end if
        end do
        end if

        C = C_temp
    end subroutine binomialCoefficient

    subroutine index_map(Sz_subspace_size, ind_1D_target, ind_i, ind_j) 
        implicit none 
        integer(8), intent(in) :: Sz_subspace_size, ind_1D_target
        integer(8), intent(out) :: ind_i, ind_j
        integer(8) :: ind_3
        integer(8) :: ind_Sz_1, ind_Sz_2, ind_tmp, loop_occ, N

        ind_tmp = ind_1D_target
        N = Sz_subspace_size
        loop_occ = 0 

        do while (ind_tmp .GT. 0)
            ind_tmp = ind_tmp - N
            N = N - 1
            loop_occ = loop_occ + 1
        end do 
        ind_i = loop_occ
        ind_j = ind_i + ind_tmp + N 

    end subroutine index_map


    subroutine Sz_subspace_choice(N_spin, Sz_choice, hash_Sz, Sz_subspace_size)
        implicit none
        integer, intent(in) :: N_spin
        double precision, intent(in) :: Sz_choice
        integer(8), intent(out) :: Sz_subspace_size !beacuse of overflow for N_spin >= 18
        integer, allocatable, intent(out) :: hash_Sz(:) !try to allocate it as small as possible to save memory
        
        integer :: N_spin_max, i, j, ind_2, Sz_choice_ind
        double precision :: Sz
        logical :: bool
        
        !double precision, allocatable   :: Sz_basis(:)
        !info
        !
        !CHARACTER*1 :: id

        N_spin_max = 2**N_spin
        
        !from all N_spin_max we should be able to predict degeneracy of restricted S_z
        !1) we need to know Sz_choice_ind from Sz_choice and N_spin_max
        !Sz_choice = N_spin_max/2      -> Sz_choice_ind = 0
        !Sz_choice = N_spin_max/2-1    -> Sz_choice_ind = 1
        !Sz_choice = N_spin_max/2-2    -> Sz_choice_ind = 2
        write(*,*) 'choosen spin= ', Sz_choice
        
        Sz_choice_ind = int( N_spin/2.0d0 - Sz_choice )
        write(*,*) 'calculated integer Sz_choice_ind = ', Sz_choice_ind
        
        !2) Sz_subspace_size = 4 here Newton symbol to be implemented = ( N_spin_max over Sz_choice_ind )
        
        !testing binomialCoefficient(n, k, C)
        call binomialCoefficient(N_spin, Sz_choice_ind, Sz_subspace_size)
        
        !Sz_subspace_size = 4 ! standard factorial calcualated recursively/ logartihmically
        write(*,*) 'Sz_subspace_size = ', Sz_subspace_size
        
        !Sz_subspace_size = 1
        
        allocate ( hash_Sz(Sz_subspace_size) )

        ! we can use e.g. btest function, which returns info if given bit is 0 (false) or 1 (true) 
        !bool_T/F=btest(number, bit_number)
        ! 0 == false => Sz=Sz+1/2, 1= true> = Sz=Sz-1/2
        ind_2 = 1
        do i = 0 , N_spin_max-1
            !basis(i) = i !here we want Sz from this i
            !write(*,*) 'basis vector i =', i+1
            Sz = 0.0d0
            do j=0, N_spin-1
                bool = btest(i, j) 
                !write(*,*) j, bool
                
                if (bool==.FALSE.) then
                    Sz=Sz+1.0d0/2.0d0
                elseif (bool==.TRUE.) then
                    Sz=Sz-1.0d0/2.0d0
                else
                    write(*,*) 'something wrong'
                    error stop
                end if                
            end do    
            
            !Sz_basis(i+1) = Sz
            if (Sz==Sz_choice) then
                    hash_Sz(ind_2) = i+1
                    ind_2 = ind_2 + 1
            end if
            
            !write(*,*) 'for basis vector i =', i, 'Sz = ', Sz
        end do
        
        write(*,*) 'internal hash_Sz size check', ind_2-1, 'vs', Sz_subspace_size
        
        !open (unit=40, file="hash_Sz_basis.dat", recl=512)
        !write(*,*) 'Summary of hash_Sz basis: '
        !do i=1,ind_2-1
        !    write(* ,*) i, hash_Sz(i)
        !    write(40,*) i, hash_Sz(i)
        !end do
        !close(40)
    end subroutine Sz_subspace_choice


    subroutine H_XXX_subspace_fill_and_diag_fast(N_spin, J_spin, Sz_choice, Sz_subspace_size, hash_Sz, e, x, m)
        use omp_lib
        use mkl_vsl
        implicit none
        !OLD VERSION OF THE FUNCTION CHECK TIME WINDOW VERSION BELOW
        
        !character(len = 12), intent(in) :: N_spin_char
        integer , dimension(8) :: start_csr, finish_csr, start_diag, finish_diag
        integer, intent(in) :: N_spin
        integer (8), intent(in) :: Sz_subspace_size
        double precision, intent(in) :: Sz_choice
        integer, allocatable :: hash_Sz(:), list_of_ind(:,:), list_of_ind_2(:,:), ia(:), ja(:), open_mp_counter(:)
        logical, allocatable :: list_of_ind_bool(:)
        double precision, intent(in) :: J_spin
        integer :: i, j, ind_i, ind_j, N_spin_max, ind_Sz_1, ind_Sz_2, ind_3, size_of_1D_list_for_sweep, &
                size_of_list, ja_val_arr_size, ind_temp, ind_temp_2, omp_id, threads_max
        double precision, allocatable :: H_full(:,:), val_arr(:)
        double precision :: norm, H_full_ij, element_value_cutoff

        double precision, allocatable, intent(out) :: x(:,:)
        integer :: fpm(128)
        double precision :: emin, emax
        integer :: m0
        double precision :: epsout
        integer :: loop
        double precision, allocatable, intent(out) :: e(:)
        integer, intent(out) :: m
        double precision, allocatable :: res(:)
        
        CHARACTER*1 :: jobz, range, uplo
        
        integer :: il, iu, ldz, liwork, lwork, info, m_eig, n

        ! lets rewrite our matlab code
        ! N=4 spin-1/2 Heisenberg XXX (Jx=Jy=Jz=J) model, J=1

        N_spin_max = 2**N_spin
        write(*,*) 'be carefull about N_spin_max max size - large integer possibility'
        write(*,*) 'test for N_spin_max', N_spin_max
        
        ! 1D list for fast sweep 
        ! changed now for a method without allocation
        size_of_list = Sz_subspace_size*(Sz_subspace_size + 1  ) / 2
        allocate( list_of_ind(size_of_list, 2) )
        allocate( list_of_ind_bool(size_of_list) )
        list_of_ind_bool = .FALSE.
        ind_3 = 1
        do ind_Sz_1 = 1        , Sz_subspace_size
        do ind_Sz_2 = ind_Sz_1 , Sz_subspace_size
            list_of_ind(ind_3,1) = ind_Sz_1
            list_of_ind(ind_3,2) = ind_Sz_2
            ind_3 = ind_3 + 1
        end do
        end do
        
        ! fast 1D sweep calculating size of ja and val_arr vectors
        write(*,*) 'OpenMP processors check'
        threads_max = omp_get_max_threads()
        write(*,*) 'OpenMP thread max num check', threads_max
        
        allocate(open_mp_counter(threads_max) )
        open_mp_counter = 0
        
            omp_id = omp_get_thread_num()
            !open_mp_counter(omp_id+1) = open_mp_counter(omp_id+1) + 1
            !write(*,*) 'OpenMP thread check', omp_id

        element_value_cutoff =  10.0d0**(-8.00d0) ! only larger elements are taken as non-zero
        !ja_val_arr_size = 0
        !$OMP PARALLEL DO PRIVATE(ind_3, omp_id, ind_Sz_1, ind_Sz_2, ind_i, ind_j, H_full_ij) &
                        SHARED(size_of_list, list_of_ind, open_mp_counter, list_of_ind_bool, hash_Sz, N_spin, J_spin, element_value_cutoff) &
                        default(none)
                        
        do ind_3 = 1, size_of_list
            omp_id = omp_get_thread_num()
            !call index_map(Sz_subspace_size, ind_3, ind_Sz_1, ind_Sz_2)
            ind_Sz_1 = list_of_ind(ind_3,1)
            ind_Sz_2 = list_of_ind(ind_3,2)
            if (ind_Sz_1 == ind_Sz_2) then
                !ja_val_arr_size = ja_val_arr_size + 1
                open_mp_counter(omp_id+1) = open_mp_counter(omp_id+1) + 1
                list_of_ind_bool(ind_3) = .TRUE.
            else if (ind_Sz_2 .GT. ind_Sz_1) then 
                ind_i = hash_Sz(ind_Sz_1)
                ind_j = hash_Sz(ind_Sz_2)
                call H_XXX_filling(N_spin, J_spin, ind_i, ind_j, H_full_ij)
                if ( abs(H_full_ij) .GT. element_value_cutoff ) then
                    !ja_val_arr_size = ja_val_arr_size + 1
                    open_mp_counter(omp_id+1) = open_mp_counter(omp_id+1) + 1
                    list_of_ind_bool(ind_3) = .TRUE.
                end if
            else
                write(*,*) 'some error in 1D list'
                end if
        end do
        !$OMP END PARALLEL DO 
        
        write(*,*) 'test of open_mp_counter'
        do i = 1, threads_max
            write(*,*) i, open_mp_counter(i)
        end do
        
        ja_val_arr_size = sum(open_mp_counter)
        !write(*,*) 'necessary ja and val_arr size = ', ja_val_arr_size
        
        
        !do ind_3 = 1, size_of_list
            !write(*,*) ind_3, list_of_ind(ind_3,1), list_of_ind(ind_3,2), list_of_ind_bool(ind_3)                    
        !end do
        
        allocate( list_of_ind_2(ja_val_arr_size, 2) )
    ! allocate( list_of_ind_bool_2(ja_val_arr_size) )
        
        ind_temp = 1
        do ind_3 = 1, size_of_list
            if( list_of_ind_bool(ind_3) ) then
                !call index_map(Sz_subspace_size, ind_3, ind_Sz_1, ind_Sz_2)
                list_of_ind_2(ind_temp, 1) =  list_of_ind(ind_3, 1)
                list_of_ind_2(ind_temp, 2) =  list_of_ind(ind_3, 2)
                !write(*,*) ind_3, list_of_ind_2(ind_temp, 1), list_of_ind_2(ind_temp, 2)
                ind_temp = ind_temp+1
            end if
        end do
        
        deallocate (list_of_ind, list_of_ind_bool)
        allocate( ia( Sz_subspace_size+1), ja(ja_val_arr_size), val_arr(ja_val_arr_size) )
        !ia and ja inside list_of_ind_2
        
        !asynchronous filling of val_arr!
        !$OMP PARALLEL DO PRIVATE(ind_3, ind_i, ind_j, H_full_ij, omp_id) &
                        SHARED(ja_val_arr_size, hash_Sz, list_of_ind_2, N_spin, J_spin, val_arr, ja) &
                        default(none)
        do ind_3 = 1, ja_val_arr_size      
            omp_id = omp_get_thread_num()
            ind_i = hash_Sz( list_of_ind_2(ind_3, 1) )
            ind_j = hash_Sz( list_of_ind_2(ind_3, 2) )
            call H_XXX_filling(N_spin, J_spin, ind_i, ind_j, H_full_ij)
            !write(*,*) 'id ', omp_id , 'ind_3', ind_3, ind_i, ind_j, H_full_ij
            val_arr(ind_3) = H_full_ij
            ja(ind_3) = list_of_ind_2(ind_3, 2)
        end do
        !$OMP END PARALLEL DO 
        
        ! loop to calculate ia()
        ind_temp = 1
        ind_temp_2 = 1
        do ind_3 = 1, Sz_subspace_size                
            ia(ind_3) = ind_temp
            DO WHILE ( list_of_ind_2(ind_temp_2, 1) == ind_3 ) 
                ind_temp = ind_temp+1
                ind_temp_2 = ind_temp_2 + 1
            END DO 
        end do
        ia(Sz_subspace_size +1) = ind_temp
        
        !write(*,*) 'ia: '
        !do ind_3 = 1, Sz_subspace_size+1
        !    write(*,*) ind_3, ia(ind_3)    
        !end do
        
        !write(*,*) 'ja and val_arr:'
        !do ind_3 = 1, ja_val_arr_size
        !    write(*,*) ind_3, ja(ind_3), val_arr(ind_3)
        !end do
    

        !FEAST diagonalization

        !n=n     ! Sets the size of the problem
        !a=non_zero_array     ! Array containing the nonzero elements of the upper triangular part of the matrix A
        call feastinit (fpm)  ! function specifying default parameters fpm of FEAST algorithm
        fpm(1) = 1
        fpm(2) = 12 !can be more, can be less
        fpm(3) = 8 !eps 10^-fpm(3)
        fpm(4) = 20 ! max number of feast loops
        fpm(5) = 0 !initial subspace
        fpm(6) = 0! stopping criterion
        fpm(7) = 5 !Error trace sigle prec stop crit
        fpm(14) = 0 ! standard use of feast
        fpm(27) = 1 !check input matrices
        fpm(28) = 1 !check if B is positive definite?         
        uplo='U' ! If uplo = 'U', a stores the upper triangular parts of A.
        emin = -20.0d0 ! The lower ... &
        emax =  20.0d0  !  and upper bounds of the interval to be searched for eigenvalues
        m0 = Sz_subspace_size !On entry, specifies the initial guess for subspace dimension to be used, 0 < m0?n. 
        !Set m0 ? m where m is the total number of eigenvalues located in the interval [emin, emax]. 
        !If the initial guess is wrong, Extended Eigensolver routines return info=3.
        n = Sz_subspace_size
        allocate( x(n,m0), e(m0), res(m0) )
        !write(*,*) 'Windows 11 new feature: feast might work only for Relase, not Debug!'
        write(*,*) 'Before dfeast_scsrev... '
        call dfeast_scsrev(uplo, n, val_arr, ia, ja, fpm, epsout, loop, emin, emax, m0, e, x, m, res, info)
        write(*,*) 'eps_out= ', epsout
        write(*,*) 'loop= ', loop
        write(*,*) ' dfeast_scsrev info=', info
        write(*,*) 'After  dfeast_scsrev... '
        
        if (info /= 0) then
        write(*,*) 'problem with  dfeast_scsrev, info=', info
        end if 

        !file_name1 = 'Eigenvalues_results_' // trim(adjustl(N_spin_char)) // '_feast.dat'
        !open (unit=1, file= trim(file_name1), recl=512)
        !write(1,*), "#Eigenvalue     Spin_z   norm" 

        !write(*,*) ' dfeast_scsrev eigenvalues found= ', m
        !do i = 1, m
         !   write(*,*) i, e(i)
        !end do

        !write(*,*) ' dfeast_scsrev eigenvec norm:'
        !do i=1,m
        !    norm = 0.0d0
         !   do j=1, n
         !       norm = norm + x(j,i)*(x(j,i))
         !   end do
         !   write(*,*) i, norm
          !  write(1,*) e(i), ',' , Sz_choice, ',', norm
            
        !end do

        !close(1)      
        !deallocate( val_arr, ia, ja, x, e, res)
        deallocate( val_arr, ia, ja, res)
        deallocate( list_of_ind_2 )
        deallocate( open_mp_counter )

    end subroutine H_XXX_subspace_fill_and_diag_fast


    subroutine H_XXX_subspace_fill_and_diag_fast_time_window(N_spin, J_spin, Sz_choice, Sz_subspace_size, hash_Sz, e, x, m)
        use omp_lib
        use mkl_vsl
        implicit none
        !character(len = 12), intent(in) :: N_spin_char
        integer, dimension(8) :: start_csr, finish_csr, start_diag, finish_diag
        integer, intent(in) :: N_spin
        integer (8), intent(in) :: Sz_subspace_size
        double precision, intent(in) :: Sz_choice
        integer, allocatable :: hash_Sz(:), list_of_ind_2(:,:), ia(:), ja(:), open_mp_counter(:) !list_of_ind(:,:) changed to 2 bytes below
        logical(1), allocatable :: list_of_ind_bool(:)
        double precision, intent(in) :: J_spin
        integer :: i, j, ind_i, ind_j, ind_ja, ind_temp, ind_temp_2, N_spin_max, size_of_1D_list_for_sweep, &
                    ja_val_arr_size, omp_id, threads_max
        integer(4), allocatable :: list_of_ind(:,:) ! Signed integer value from -32,768 to 32,767
        integer(8) :: ind_Sz_1, ind_Sz_2, ind_3
        integer(8) :: size_of_list !8 byte = 64 bit Signed integer value from -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807
        double precision, allocatable :: H_full(:,:), val_arr(:)
        double precision :: norm, H_full_ij, element_value_cutoff, e_min_reg, e_max_reg
        double precision, allocatable, intent(out) :: x(:,:)
        integer :: fpm(128)
        double precision :: emin, emax
        integer :: m0
        double precision :: epsout, t1, t2
        integer :: loop
        double precision, allocatable, intent(out) :: e(:)
        integer, intent(out) :: m
        double precision, allocatable :: res(:)
        
        CHARACTER*1 :: jobz, range, uplo
        
        integer :: il, iu, ldz, liwork, lwork, info, m_eig, n

        ! lets rewrite our matlab code
        ! N=4 spin-1/2 Heisenberg XXX (Jx=Jy=Jz=J) model, J=1
        call date_and_time(VALUES=start_csr)
        t1 = omp_get_wtime()

        N_spin_max = 2**N_spin
        write(*,*) 'be carefull about N_spin_max max size - large integer possibility'
        write(*,*) 'test for N_spin_max', N_spin_max
        
        ! 1D list for fast sweep 
        ! changed now for a method without allocation
        size_of_list = Sz_subspace_size*(Sz_subspace_size + 1  ) / 2

        !write(*,*) 'Before the lists allocation for 1d sweep'
        write(*,*) 'size_of_list = ', size_of_list
        !allocate( list_of_ind(size_of_list, 2) )
        !write(*,*) 'After the list_of_ind allocation for 1d sweep'
        write(*,*) 'Before the lists bool allocation for 1d sweep'
        allocate( list_of_ind_bool(size_of_list) )
        write(*,*) 'After the list_of_ind bool allocation for 1d sweep'

        !list_of_ind_bool = .FALSE.
        !ind_3 = 1
        !do ind_Sz_1 = 1        , Sz_subspace_size
        !do ind_Sz_2 = ind_Sz_1 , Sz_subspace_size
        !    list_of_ind(ind_3,1) = ind_Sz_1
        !    list_of_ind(ind_3,2) = ind_Sz_2
        !    ind_3 = ind_3 + 1
        !end do
        !end do
        
        ! fast 1D sweep calculating size of ja and val_arr vectors
        write(*,*) 'OpenMP processors check'
        threads_max = omp_get_max_threads()
        write(*,*) 'OpenMP thread max num check', threads_max
        
        allocate(open_mp_counter(threads_max) )
        open_mp_counter = 0
        
            omp_id = omp_get_thread_num()
            !open_mp_counter(omp_id+1) = open_mp_counter(omp_id+1) + 1
            !write(*,*) 'OpenMP thread check', omp_id

        element_value_cutoff =  10.0d0**(-8.00d0) ! only larger elements are taken as non-zero
        !ja_val_arr_size = 0
        !$OMP PARALLEL DO PRIVATE(ind_3, omp_id, ind_Sz_1, ind_Sz_2, ind_i, ind_j, H_full_ij) &
                        SHARED(size_of_list, list_of_ind, open_mp_counter, list_of_ind_bool, hash_Sz, N_spin, J_spin, element_value_cutoff, Sz_subspace_size) &
                        default(none)
                        
        do ind_3 = 1, size_of_list
            omp_id = omp_get_thread_num()
            call index_map(Sz_subspace_size, ind_3, ind_Sz_1, ind_Sz_2)
            !ind_Sz_1 = list_of_ind(ind_3,1)
            !ind_Sz_2 = list_of_ind(ind_3,2)
            if (ind_Sz_1 == ind_Sz_2) then
                !ja_val_arr_size = ja_val_arr_size + 1
                open_mp_counter(omp_id+1) = open_mp_counter(omp_id+1) + 1
                list_of_ind_bool(ind_3) = .TRUE.
            else if (ind_Sz_2 .GT. ind_Sz_1) then 
                ind_i = hash_Sz(ind_Sz_1)
                ind_j = hash_Sz(ind_Sz_2)
                call H_XXX_filling(N_spin, J_spin, ind_i, ind_j, H_full_ij)
                if ( abs(H_full_ij) .GT. element_value_cutoff ) then
                    !ja_val_arr_size = ja_val_arr_size + 1
                    open_mp_counter(omp_id+1) = open_mp_counter(omp_id+1) + 1
                    list_of_ind_bool(ind_3) = .TRUE.
                end if
            else
                write(*,*) 'some error in 1D list'
            end if
            !if (list_of_ind_bool(ind_3)) then
            !    open_mp_counter(omp_id+1) = open_mp_counter(omp_id+1) + 1
            !end if
        end do
        !$OMP END PARALLEL DO 
        
        write(*,*) 'test of open_mp_counter'
        do i = 1, threads_max
            write(*,*) i, open_mp_counter(i)
        end do
        
        ja_val_arr_size = sum(open_mp_counter)
        write(*,*) 'necessary ja and val_arr size = ', ja_val_arr_size
        
        
        !do ind_3 = 1, size_of_list
            !write(*,*) ind_3, list_of_ind(ind_3,1), list_of_ind(ind_3,2), list_of_ind_bool(ind_3)                    
        !end do
        write(*,*) 'Before the lists allocation for 2nd sweep'
        allocate( list_of_ind_2(ja_val_arr_size, 2) )
    ! allocate( list_of_ind_bool_2(ja_val_arr_size) )
        
        ind_temp = 1
        do ind_3 = 1, size_of_list
            if( list_of_ind_bool(ind_3) ) then
                call index_map(Sz_subspace_size, ind_3, ind_Sz_1, ind_Sz_2)
                list_of_ind_2(ind_temp, 1) = ind_Sz_1 !list_of_ind(ind_3, 1)
                list_of_ind_2(ind_temp, 2) = ind_Sz_2 !list_of_ind(ind_3, 2)
                !write(*,*) ind_3, list_of_ind_2(ind_temp, 1), list_of_ind_2(ind_temp, 2)
                ind_temp = ind_temp+1
            end if
        end do
        
        deallocate (list_of_ind_bool)
        write(*,*) 'Before the csr vectors allocation'
        allocate( ia( Sz_subspace_size+1), ja(ja_val_arr_size), val_arr(ja_val_arr_size) )
        !ia and ja inside list_of_ind_2
        
        !asynchronous filling of val_arr!
        !$OMP PARALLEL DO PRIVATE(ind_ja, ind_i, ind_j, H_full_ij, omp_id) &
                        SHARED(ja_val_arr_size, hash_Sz, list_of_ind_2, N_spin, J_spin, val_arr, ja) &
                        default(none)
        do ind_ja = 1, ja_val_arr_size      
            omp_id = omp_get_thread_num()
            ind_i = hash_Sz( list_of_ind_2(ind_ja, 1) )
            ind_j = hash_Sz( list_of_ind_2(ind_ja, 2) )
            call H_XXX_filling(N_spin, J_spin, ind_i, ind_j, H_full_ij)
            !write(*,*) 'id ', omp_id , 'ind_3', ind_3, ind_i, ind_j, H_full_ij
            val_arr(ind_ja) = H_full_ij
            ja(ind_ja) = list_of_ind_2(ind_ja, 2)
        end do
        !$OMP END PARALLEL DO 
        
        ! loop to calculate ia()
        ind_temp = 1
        ind_temp_2 = 1
        do ind_3 = 1, Sz_subspace_size                
            ia(ind_3) = ind_temp
            DO WHILE ( list_of_ind_2(ind_temp_2, 1) == ind_3 ) 
                ind_temp = ind_temp+1
                ind_temp_2 = ind_temp_2 + 1
            END DO 
        end do
        ia(Sz_subspace_size +1) = ind_temp
        
        !write(*,*) 'ia: '
        !do ind_3 = 1, Sz_subspace_size+1
        !    write(*,*) ind_3, ia(ind_3)    
        !end do
        
        !write(*,*) 'ja and val_arr:'
        !do ind_3 = 1, ja_val_arr_size
        !    write(*,*) ind_3, ja(ind_3), val_arr(ind_3)
        !end do
        call date_and_time(VALUES=finish_csr) !end time counter
        t2 = omp_get_wtime()

        !old example for date_and_time
        !write(*,*) 'Csr filling time = ', &
            !(((finish_csr(5) - start_csr(5))*3600 + &
            !(finish_csr(6) - start_csr(6))*60 + &
            !(finish_csr(7) - start_csr(7)))*1000 + &
            !(finish_csr(8) - start_csr(8)))/1000, "seconds."

        print '("Csr filling time = ",f," seconds.")',t2-t1
        write(3,*) "Csr filling time = ", t2-t1 ," seconds."

        !Calculation of energy bounds for the chain problem according to the previous estimations: 
        e_min_reg = -0.4465d0 * N_spin + 0.1801d0
        e_max_reg = -0.49773d0 * N_spin + 2.10035d0

        write(*,*) "Calculated lower bound: ", e_min_reg
        write(*,*) "Calculated upper bound: ", e_max_reg
        !FEAST diagonalization

        !n=n     ! Sets the size of the problem
        !a=non_zero_array     ! Array containing the nonzero elements of the upper triangular part of the matrix A
        call feastinit (fpm)  ! function specifying default parameters fpm of FEAST algorithm
        fpm(1) = 1
        fpm(2) = 12 !can be more, can be less
        fpm(3) = 8 !eps 10^-fpm(3)
        fpm(4) = 20 ! max number of feast loops
        fpm(5) = 0 !initial subspace
        fpm(6) = 0! stopping criterion
        fpm(7) = 5 !Error trace sigle prec stop crit
        fpm(14) = 0 ! standard use of feast
        fpm(27) = 1 !check input matrices
        fpm(28) = 1 !check if B is positive definite?         
        uplo='U' ! If uplo = 'U', a stores the upper triangular parts of A.
        emin = e_min_reg! The lower ... &
        emax = e_max_reg  !  and upper bounds of the interval to be searched for eigenvalues
        m0 = 12 !On entry, specifies the initial guess for subspace dimension to be used, 0 < m0?n. 
        !Set m0 ? m where m is the total number of eigenvalues located in the interval [emin, emax]. 
        !If the initial guess is wrong, Extended Eigensolver routines return info=3.
        n = Sz_subspace_size
        allocate( x(n,m0), e(m0), res(m0) )
        !write(*,*) 'Windows 11 new feature: feast might work only for Relase, not Debug!'
        write(*,*) 'Before dfeast_scsrev... '

        call date_and_time(VALUES=start_diag)
        t1 = omp_get_wtime()
        call dfeast_scsrev(uplo, n, val_arr, ia, ja, fpm, epsout, loop, emin, emax, m0, e, x, m, res, info)
        t2 = omp_get_wtime()
        call date_and_time(VALUES=finish_diag)
        
        write(*,*) 'eps_out= ', epsout
        write(*,*) 'loop= ', loop
        write(*,*) ' dfeast_scsrev info=', info
        write(*,*) 'After  dfeast_scsrev... '
        
        if (info /= 0) then
        write(*,*) 'problem with  dfeast_scsrev, info=', info
        end if 

        print '("Diagonalisation time = ",f," seconds.")',t2-t1
        write(3, *) "Diagonalisation time = ", t2-t1 ," seconds."

        !file_name1 = 'Eigenvalues_results_' // trim(adjustl(N_spin_char)) // '_feast.dat'
        !open (unit=1, file= trim(file_name1), recl=512)
        !write(1,*), "#Eigenvalue     Spin_z   norm" 

        !write(*,*) ' dfeast_scsrev eigenvalues found= ', m
        !do i = 1, m
         !   write(*,*) i, e(i)
        !end do

        !write(*,*) ' dfeast_scsrev eigenvec norm:'
        !do i=1,m
        !    norm = 0.0d0
         !   do j=1, n
         !       norm = norm + x(j,i)*(x(j,i))
         !   end do
         !   write(*,*) i, norm
          !  write(1,*) e(i), ',' , Sz_choice, ',', norm
            
        !end do

        !close(1)      
        !deallocate( val_arr, ia, ja, x, e, res)
        deallocate( val_arr, ia, ja, res)
        deallocate( list_of_ind_2 )
        deallocate( open_mp_counter )

    end subroutine H_XXX_subspace_fill_and_diag_fast_time_window


    subroutine H_XXX_filling(N_spin, J_spin, i, j, H_full_ij)
        implicit none
        integer, intent(in) :: N_spin, i, j
        double precision, intent(in) :: J_spin
        double precision, intent(out) :: H_full_ij
        integer :: N_spin_max  
        double precision :: Sp_site_ind_Sm_site_ind_p1_ij, Sm_site_ind_Sp_site_ind_p1_ij, Sz_site_ind_Sz_site_ind_p1_ij
        integer :: site_ind, m, n, p, q, r, s
        integer :: ind_i, ind_j, a_i, a_j, b_i, b_j, c_i, c_j
        double precision :: A_val, B_val, C_val

        ! N spin-1/2 Heisenberg XXX (Jx=Jy=Jz=J) model, J=1
        N_spin_max = 2**N_spin

        H_full_ij = 0.0d0
                           
        do site_ind = 1, (N_spin-1) !here change for searching through the adjacency matrix for graphs
            m = 2**(site_ind-1)
            n = m
            p = 4 ! N_spin independent because this referes to S+S- operator for spin 1/2 
            q = 4
            r = 2**(N_spin-(site_ind+1))
            s = r
            !write(*,*) site_ind, m, n, p, q, r, s
        
            Sp_site_ind_Sm_site_ind_p1_ij = 0.0d0
            Sm_site_ind_Sp_site_ind_p1_ij = 0.0d0
            Sz_site_ind_Sz_site_ind_p1_ij = 0.0d0
            
            !do ind_i = 1, N_spin_max
            !do ind_j = 1, N_spin_max
            ind_i = i
            ind_j = j
                !write(*,*) 'site_ind, ind_i, ind_j = ',  site_ind, ind_i, ind_j
                !i = ind_i-1
                !j = ind_j-1
                a_i = FLOOR( (ind_i-1.0d0)/( (p*r)+0.0d0 ) )
                a_j = FLOOR( (ind_j-1.0d0)/( (q*s)+0.0d0 ) )
                b_i = MOD(FLOOR( (ind_i-1.0d0)/( r+0.0d0 ) ), p)
                b_j = MOD(FLOOR( (ind_j-1.0d0)/( s+0.0d0 ) ), q)
                c_i = MOD(ind_i-1, r)
                c_j = MOD(ind_j-1, s)
                                
                A_val = 0.0d0
                B_val = 0.0d0
                C_val = 0.0d0
            
                if (a_i == a_j) then
                    A_val = 1.0d0
                else
                    A_val = 0.0d0
                end if
            
                if (c_i == c_j) then
                    C_val = 1.0d0
                else
                    C_val = 0.0d0
                end if
            
                ! For S+S-:
                if (b_i == 1 .AND. b_j == 2) then
                    B_val = 1.0d0
                else
                    B_val = 0.0d0
                end if
                Sp_site_ind_Sm_site_ind_p1_ij = A_val*B_val*C_val
            
                ! For S-S+:
                if (b_i == 2 .AND. b_j == 1) then
                    B_val = 1.0d0
                else
                    B_val = 0.0d0
                end if                    
                Sm_site_ind_Sp_site_ind_p1_ij = A_val*B_val*C_val
            
                ! For SzSz:
                if (b_i == 0 .AND. b_j == 0) then
                    B_val = 0.250d0
                else if (b_i == 1 .AND. b_j == 1) then
                    B_val = -0.250d0
                else if (b_i == 2 .AND. b_j == 2) then
                    B_val = -0.250d0
                else if (b_i == 3 .AND. b_j == 3) then
                    B_val = 0.250d0
                else
                    B_val = 0.0d0
                end if

                Sz_site_ind_Sz_site_ind_p1_ij= A_val*B_val*C_val
            
            !end do
            !end do
        
            H_full_ij = H_full_ij + 1.0d0/2.0d0 *(Sp_site_ind_Sm_site_ind_p1_ij+Sm_site_ind_Sp_site_ind_p1_ij) + Sz_site_ind_Sz_site_ind_p1_ij
            
        end do


    end subroutine H_XXX_filling


    subroutine omp_mkl_small_test()
        use omp_lib
        use mkl_vsl
        implicit none

        integer :: i, j, omp_id, brng, rand_method, rand_seed, errcode, no_of_rand_num
        real(kind=8) rand_start, rand_end
        real(kind=8), allocatable :: rand_num(:)
        type (VSL_STREAM_STATE) :: stream(1)

        CHARACTER*1 :: jobz, range, uplo
        integer :: N_mat, il, iu, ldz, liwork, m, lwork, lrwork, info
        integer, allocatable :: iwork(:), isuppz(:)
        double precision :: vl, vu, abstol
        double complex, allocatable :: H_test(:,:), work_complex(:), eigen_vec(:,:)
        double precision, allocatable :: rwork(:), w(:)
        double complex :: norm

        double complex, allocatable  :: inv_H_test(:,:)
        integer, allocatable :: ipv(:)

        write(*,*) '----------- OMP MKL small matrix test:-----------'
        write(*,*) " "

        !   0 - part checking OpenMP + IntelMKL (oneAPI) random numbers/diag/inversion/
        write(*,*) 'OpenMP processors check'
        !$OMP PARALLEL PRIVATE(omp_id)
        omp_id = omp_get_thread_num()
        write(*,*) 'OpenMP thread check', omp_id
        !$OMP END PARALLEL



        write(*,*) 'Math Kernel Library random numbers check'
        ! *** random number generator parameters *** !
        brng   = VSL_BRNG_MT2203
        rand_method = VSL_RNG_METHOD_UNIFORM_STD
        rand_start = 0.0
        rand_end = 1.0
        rand_seed = 1
        no_of_rand_num = 10
        allocate(rand_num(no_of_rand_num))
        errcode = vslnewstream(stream(1), brng, rand_seed)
        write(*,*) 'random stream init code: ', errcode
        errcode=vdrnguniform( rand_method, stream(1), no_of_rand_num, rand_num, rand_start, rand_end )
        write(*,*) 'random numbers error code: ', errcode
        do i = 1, no_of_rand_num
            write(*,*) i, rand_num(i)
        end do
        deallocate(rand_num)
        errcode=vsldeletestream(stream(1))
        write(*,*) 'random stream kill code: ', errcode
        !end do
        write(*,*) 'Small Hermitean matrix diagonalization test'
        jobz  = 'V' !Compute eigenvalues and eigenvectors.
        range = 'I' !IL-th through IU-th eigenvalues will be found
        uplo  = 'U' !Upper triangle of A is stored;
        N_mat = 3 ! The order of the matrix
        vl = -30.0d0 ! lower bound of eigenvalues (for range='V')
        vu = 30.0d0  ! upper bound of eigenvalues (for range='V')
        il=1  !the index of the smallest eigenvalue to be returned.
        iu=N_mat !the index of the largest eigenvalue to be returned.
        abstol=10**(-10.0d0)
        ldz=N_mat
        liwork= 10*N_mat
        lwork=2*N_mat
        lrwork=24*N_mat
        allocate (H_test(N_mat,N_mat))
        H_test(1,1) = dcmplx(1.0d0, 0.0d0)
        H_test(1,2) = dcmplx(1.0d0, 1.0d0)
        H_test(1,3) = dcmplx(2.0d0, 2.0d0)
        H_test(2,1) = dcmplx(1.0d0,-1.0d0)
        H_test(2,2) = dcmplx(2.0d0, 0.0d0)
        H_test(2,3) = dcmplx(3.0d0, 3.0d0)
        H_test(3,1) = dcmplx(2.0d0,-2.0d0)
        H_test(3,2) = dcmplx(3.0d0,-3.0d0)
        H_test(3,3) = dcmplx(3.0d0, 0.0d0)
        allocate ( eigen_vec(N_mat,N_mat), work_complex(lwork),iwork(liwork),w(N_mat),isuppz(2*N_mat), rwork(lrwork)  )
        call zheevr(jobz, range, uplo, N_mat, H_test, N_mat, vl, vu, il, iu, abstol, m, w, eigen_vec, ldz, isuppz, work_complex, lwork, rwork, lrwork, iwork, liwork, info)
        write(*,*) "general zheevr info:", info
        write(*,*) lwork, " vs optimal lwork:", work_complex(1)
        write(*,*) "number of eigeval found:", m
        do i= 1, m
            write(*,*) i, 'th eigenval is', w(i)
        end do
        write(*,*) 'compare with Matlab: -2.4801476299237900'
        write(*,*) 'compare with Matlab:  0.5056116166572769'
        write(*,*) 'compare with Matlab:  7.9745360132665200'



        write(*,*) 'eigenvec norm test:'
        do i=1,m
            norm = dcmplx(0.0d0, 0.0d0)
            do j=1,N_mat
                norm = norm + eigen_vec(j,i)*conjg(eigen_vec(j,i))
            end do
            write(*,*) i, norm
        end do
        deallocate ( eigen_vec, work_complex, iwork, w, isuppz, rwork  )
        !
        write(*,*) 'Small Hermitean matrix inversion test'
        H_test(1,1) = dcmplx(1.0d0, 0.0d0)
        H_test(1,2) = dcmplx(1.0d0, 1.0d0)
        H_test(1,3) = dcmplx(2.0d0, 2.0d0)
        H_test(2,1) = dcmplx(1.0d0,-1.0d0)
        H_test(2,2) = dcmplx(2.0d0, 0.0d0)
        H_test(2,3) = dcmplx(3.0d0, 3.0d0)
        H_test(3,1) = dcmplx(2.0d0,-2.0d0)
        H_test(3,2) = dcmplx(3.0d0,-3.0d0)
        H_test(3,3) = dcmplx(3.0d0, 0.0d0)
        lwork = N_mat
        allocate(inv_H_test(N_mat,N_mat), ipv(N_mat), work_complex(lwork) )
        call zgetrf(N_mat, N_mat, H_test, N_mat, ipv, info ) !number of operations: (8/3)*N_mat^3
        write(*,*) 'blas LU info ', info
        call zgetri(N_mat, H_test, N_mat, ipv, work_complex, N_mat, info )
        !number of operations: (16/3)*M^3
        write(*,*) 'blas inversion info ', info
        write(*,*) lwork, 'inversion optimal lwork ', work_complex(1)
        do i = 1, N_mat
            do j = 1, N_mat
                write(*,*) i, j, H_test(i,j)
            end do
        end do
        write(*,*) 'inverted matrix should be:'
        write(*,*) ' 1.2+0.0i, -0.9+0.3i,  0.4-0.2i'
        write(*,*) '-0.9-0.3i,  0.5+0.0i, -0.1+0.3i'
        write(*,*) ' 0.4+0.2i, -0.1-0.3i,  0.0+0.0i'
        deallocate(H_test, ipv, work_complex)

        write(*,*) '----------- END OMP MKL test -----------'
        write(*,*) " "

    end subroutine omp_mkl_small_test

    subroutine sparse_zfeast_test()

        implicit none
        double complex, allocatable :: test_matrix(:,:)
        CHARACTER*1 :: jobz, uplo            
        integer :: i, n, lda, lwork, info
        double complex, allocatable :: rwork(:)
        double complex, allocatable ::  work(:,:)
        double complex, allocatable :: values_array(:)
        integer, allocatable :: ia(:), ja(:)
        integer, dimension(64) :: fpm
        double precision :: emin, emax
        integer :: m0
        double precision :: epsout
        integer :: loop
        double precision, allocatable :: e(:), eigenvalues(:), x(:,:)
        integer :: m
        double precision, allocatable :: res(:)
        
        write(*,*) '----------- Complex FEAST test START small matrix: -----------'
        write(*,*) " "

        n = 5
        allocate (test_matrix(n,n))
        test_matrix = dcmplx(0.0d0, 0.0d0)
        test_matrix(1,1) = dcmplx(1.0d0, 0.0d0)
        test_matrix(2,2) = dcmplx(0.0d0, 0.0d0)
        test_matrix(3,3) = dcmplx(2.0d0, 0.0d0)
        test_matrix(4,4) = dcmplx(0.0d0, 0.0d0)
        test_matrix(5,5) = dcmplx(3.0d0, 0.0d0)
        test_matrix(1,4) = dcmplx(5.0d0, 0.0d0)
        test_matrix(2,3) = dcmplx(6.0d0, 0.0d0)
        test_matrix(3,5) = dcmplx(7.0d0, 0.0d0)
        
        !standard diagonalziation
        jobz='V' !N - only eigenvalues, V - eigenvalues + eigenvectors
        uplo='U' !upper hermitean matrix part passed to routine
        n=5
        lda=n
        lwork=2*n-1  
        allocate(work(lwork,lwork),rwork(3*n-2),eigenvalues(n))
        
        call zheev(jobz, uplo, n, test_matrix, lda, eigenvalues, work, lwork, rwork, info)
        if (info /= 0) then
            write(*,*) 'problem with zheev, info=', info
        end if 
            
        do i=1,n
            write(*,*) i, eigenvalues(i) 
        enddo     
        deallocate(work,rwork,eigenvalues)

        write(*,*) 'Zheev diagonalization finished, FEAST start'

        !FEAST diagonalization
        ! we need to store diagonal zeros as well !!!
        allocate( values_array(8), ia(6), ja(8) )
        
        values_array(1) = dcmplx(1.0d0, 0.0d0)
        values_array(2) = dcmplx(5.0d0, 0.0d0)
        values_array(3) = dcmplx(0.0d0, 0.0d0)
        values_array(4) = dcmplx(6.0d0, 0.0d0)
        values_array(5) = dcmplx(2.0d0, 0.0d0)
        values_array(6) = dcmplx(7.0d0, 0.0d0)
        values_array(7) = dcmplx(0.0d0, 0.0d0)
        values_array(8) = dcmplx(3.0d0, 0.0d0)
        
        ja(1) = 1
        ja(2) = 4
        ja(3) = 2
        ja(4) = 3
        ja(5) = 3
        ja(6) = 5
        ja(7) = 4
        ja(8) = 5
        
        ia(1) = 1
        ia(2) = 3
        ia(3) = 5
        ia(4) = 7
        ia(5) = 8
        ia(6) = 9 !no of non-zero +1 !!!
        
        
        !n=n     ! Sets the size of the problem
        !a=non_zero_array     ! Array containing the nonzero elements of the upper triangular part of the matrix A
        call feastinit(fpm)  ! function specifying default parameters fpm of FEAST algorithm
        fpm(1) = 1 !if 1 print runtime comments
        !fpm(2) = 12 !can be more, can be less
        !fpm(3) = 8 !eps 10^-fpm(3)
        !fpm(4) = 20 ! max number of feast loops
        !fpm(5) = 0 !initial subspace
        !fpm(6) = 0! stopping criterion
        !fpm(7) = 5 !Error trace sigle prec stop crit
        !fpm(14) = 0! standard use of feast
        !fpm(27) = 1 !check input matrices
        !fpm(28) = 1 !check if B is positive definite?    
        !fpm(13) = 0     
        uplo='U' ! If uplo = 'U', a stores the upper triangular parts of A.
        emin = -20.0d0 ! The lower ... &
        emax =  20.0d0  !  and upper bounds of the interval to be searched for eigenvalues
        m0 = 5 !On entry, specifies the initial guess for subspace dimension to be used, 0 < m0≤n. 
        !Set m0 ≥ m where m is the total number of eigenvalues located in the interval [emin, emax]. 
        !If the initial guess is wrong, Extended Eigensolver routines return info=3.
        n = 5
        allocate( x(n,m0), e(m0), res(m0) )
        !write(*,*) 'Windows 11 new feature: feast might work only for Relase, not Debug!'

        !zfeast_hcsrev(UPLO,N,A,IA,JA,fpm,epsout,loop,Emin,Emax,M0,E,X,M,res,info)

        write(*,*) 'Before zfeast_hcsrev... '
        call zfeast_hcsrev(uplo, n, values_array, ia, ja, fpm, epsout, loop, emin, emax, m0, e, x, m, res, info)
        write(*,*) 'eps_out= ', epsout
        write(*,*) 'loop= ', loop
        write(*,*) 'zfeast_hcsrev info=', info
        write(*,*) 'After zfeast_hcsrev... '
            
        if (info /= 0) then
        write(*,*) 'problem with zfeast_hcsrev, info=', info
        end if 
        
        write(*,*) 'no of eigenvalues found= ', m
        do i = 1, m
            write(*,*) i, e(i)
        end do
                    
        deallocate(test_matrix, values_array, ia, ja, x, e, res)

        write(*,*) '----------- Complex FEAST test END: -----------'
        write(*,*) " "

    end subroutine sparse_zfeast_test

    subroutine sparse_dfeast_test()
        
        implicit none
        double precision, allocatable :: test_matrix(:,:)
        CHARACTER*1 :: jobz, uplo            
        integer :: i, n, lda, lwork, info
        double precision, allocatable :: eigenvalues(:), rwork(:)
        double precision, allocatable ::  work(:,:)
        double precision, allocatable :: values_array(:), x(:,:)
        integer, allocatable :: ia(:), ja(:)
        integer :: fpm(128)
        double precision :: emin, emax
        integer :: m0
        double precision :: epsout
        integer :: loop
        double precision, allocatable :: e(:)
        integer:: m
        double precision, allocatable :: res(:)
        
        write(*,*) '----------- Double precision FEAST test START small matrix: -----------'
        write(*,*) " "

        n = 5
        allocate (test_matrix(n,n))
        test_matrix = 0.0d0
        test_matrix(1,1) = 1.0d0
        test_matrix(2,2) = 0.0d0
        test_matrix(3,3) = 2.0d0
        test_matrix(4,4) = 0.0d0
        test_matrix(5,5) = 3.0d0
        test_matrix(1,4) = 5.0d0
        test_matrix(2,3) = 6.0d0
        test_matrix(3,5) = 7.0d0
        
        !standard diagonalziation
        jobz='V' !N - only eigenvalues, V - eigenvalues + eigenvectors
        uplo='U' !upper hermitean matrix part passed to routine
        n=5
        lda=n
        lwork=3*n-1  
        allocate(work(lwork,lwork),rwork(3*n-2),eigenvalues(n))
        !call dsyev(jobz, uplo, n, a, lda, w, work, lwork, info)
        call dsyev(jobz, uplo, n, test_matrix, lda, eigenvalues, work, lwork, info)
        if (info /= 0) then
            write(*,*) 'problem with dsyev, info=', info
        end if 
            
        do i=1,n
            write(*,*) i, eigenvalues(i) 
        enddo     
        deallocate(work,rwork,eigenvalues)

        write(*,*) 'dsyev diagonalization finished, FEAST start'

        !FEAST diagonalization
        ! we need to store diagonal zeros as well !!!
        allocate( values_array(8), ia(6), ja(8) )
        
        values_array(1) = 1.0d0
        values_array(2) = 5.0d0
        values_array(3) = 0.0d0
        values_array(4) = 6.0d0
        values_array(5) = 2.0d0
        values_array(6) = 7.0d0
        values_array(7) = 0.0d0
        values_array(8) = 3.0d0
        
        ja(1) = 1
        ja(2) = 4
        ja(3) = 2
        ja(4) = 3
        ja(5) = 3
        ja(6) = 5
        ja(7) = 4
        ja(8) = 5
        
        ia(1) = 1
        ia(2) = 3
        ia(3) = 5
        ia(4) = 7
        ia(5) = 8
        ia(6) = 9 !no of non-zero +1 !!!
        
        
        !n=n     ! Sets the size of the problem
        !a=non_zero_array     ! Array containing the nonzero elements of the upper triangular part of the matrix A
        call feastinit (fpm)  ! function specifying default parameters fpm of FEAST algorithm
        fpm(1) = 1
        fpm(2) = 12 !can be more, can be less
        fpm(3) = 8 !eps 10^-fpm(3)
        fpm(4) = 20 ! max number of feast loops
        fpm(5) = 0 !initial subspace
        fpm(6) = 0! stopping criterion
        fpm(7) = 5 !Error trace sigle prec stop crit
        fpm(14) = 0 ! standard use of feast
        fpm(27) = 1 !check input matrices
        fpm(28) = 1 !check if B is positive definite?         
        uplo='U' ! If uplo = 'U', a stores the upper triangular parts of A.
        emin = -20.0d0 ! The lower ... &
        emax =  20.0d0  !  and upper bounds of the interval to be searched for eigenvalues
        m0 = 5 !On entry, specifies the initial guess for subspace dimension to be used, 0 < m0≤n. 
        !Set m0 ≥ m where m is the total number of eigenvalues located in the interval [emin, emax]. 
        !If the initial guess is wrong, Extended Eigensolver routines return info=3.
        n = 5
        allocate( x(n,m0), e(m0), res(m0) )
        !write(*,*) 'Windows 11 new feature: feast might work only for Relase, not Debug!'
        write(*,*) 'Before dfeast_scsrev... '
        call dfeast_scsrev(uplo, n, values_array, ia, ja, fpm, epsout, loop, emin, emax, m0, e, x, m, res, info)
        write(*,*) 'eps_out= ', epsout
        write(*,*) 'loop= ', loop
        write(*,*) 'dfeast_scsrev info=', info
        write(*,*) 'After dfeast_scsrev... '
            
        if (info /= 0) then
        write(*,*) 'problem with dfeast_scsrev, info=', info
        end if 
        
        write(*,*) 'no of eigenvalues found= ', m
        do i = 1, m
            write(*,*) i, e(i)
        end do
                    
        deallocate(test_matrix, values_array, ia, ja, x, e, res)

        write(*,*) '----------- Double precision FEAST test END -----------'
        write(*,*) " "

    end subroutine sparse_dfeast_test
end module spin_systems

program spin_code
    use spin_systems
    use omp_lib
    implicit none

    integer :: N_spin, N_spin_test, N_spin_max, no_of_nonzero, size_of_sub_A, size_of_sub_B, i,j, k, target_sz_spin, number_of_eigen, size_of_eigenvalues, m
    integer (8) :: Sz_subspace_size
    double precision :: J_spin, entropy_value, eigen_value_check, e_up, e_down, norm
    double precision :: start, finish, finish_one_spin, start_one_spin, start_csr, finish_csr, start_diag, finish_diag, Sz_choice
    character(len = 12) :: N_spin_char, J_spin_char
    character(len=53) :: dec2bin, spin_basis, file_name1, file_name2, file_name3, file_name4
    integer, allocatable :: Sz_basis(:), hash_Sz(:), hash(:), indices_Sz_basis_sorted(:), basis_vector(:,:), basis_rho_target(:,:), new_basis(:,:)
    double precision, allocatable :: target_sz(:), eigen_values(:), eigen_vectors(:,:), rho_reduced(:,:), e(:), x(:,:)

    ! OPIS DO POPRAWY! 
    ! Version 20.07 - dodajemy tworzenie bazy i liczenie entropii
    ! tworzenie bazy i s_z działa
    ! macierz permutacji H działa 
    ! test dla obrotu H n = 4 działa 

    ! do zrobienia
    ! zapis eigenvectora do pliku - podział na podpliki 

    ! TO DO LATER
    ! Zacznij ciac odpowiednie wektory do S_z , pytanie czy to w fortranie czy nie? 
    ! chyba w fotranie + diagonalizacja za pomocą feasta blockowych macierzy 
    ! w pythonie robisz to tnac pełną macierz H
    ! pytanie do Pawła: jak brać wycinek macierzy w H blockowym do feasta (numer eigenvalues i okno energetyczne)
    ! net cdf - binarna baza danych

    !19.12 - wprowadzamy openMP
    !od teraz wersja kodu Maćka 
    ! entropie liczymy w pythonie 

    !11.03/2024 - powrót do kodu :)

    If(command_argument_count().NE.2) Then
        write(*,*)'Error, Only N (integer) and J (double precision) is required, program stopped two many or parameter is missing'
        stop 
    endif 

    call get_command_argument(1, N_spin_char)
    call get_command_argument(2, J_spin_char)

    read(N_spin_char, *) N_spin
    read(J_spin_char, *) J_spin

    !N_spin = 4
    !J_spin = 1 

    write(*,*) 'Calculation of Heisenberg chain for N =', N_spin , 'and J = ', J_spin 
    write(*,*) ' '

    ! call mmm_csr_test()
    call omp_mkl_small_test()
    ! call test_permutation_H_for_4_sites()
    ! call sparse_zfeast_test()
    call sparse_dfeast_test()

    start = omp_get_wtime()
    write(*,*) '------- START Heisenberg Program -------'
    N_spin_max = 2**N_spin 
    allocate(target_sz(N_spin-1))

    !file_name1 = 'Test_Eigenvalues_results_feast.dat'
    file_name2 = 'Eigenvalues_results_' // trim(adjustl(N_spin_char)) // '_feast.dat'
    file_name3 = 'Cpu_Time_details_' // trim(adjustl(N_spin_char)) // '.dat'
    file_name4 = 'Eigenvectors_results_' // trim(adjustl(N_spin_char)) // '_feast.dat'

    !open (unit=1, file= trim(file_name1), recl=512)
    open (unit=2, file= trim(file_name2), recl=512)
    open (unit=3, file= trim(file_name3), recl=512)
    open (unit=4, file= trim(file_name4), recl=512)
    !write(1,*), "#Eigenvalue     Spin_z   norm" 
    write(2,*), "#Eigenvalue     Spin_z   norm" 
    write(3,*), "#Time needed for each process"
    write(4, *) "#Eigenvectors results"
    
    ! generate a target_sz array storing all possible Sz values for N_spin 
    
    do i = 1, N_spin - 1
            target_sz(i) = N_spin/2.0d0 - i
    end do 


    print *, "All combination of S_z spin", target_sz
    write(*,*) " "
    
    !write(*,*) '------- START Heisenberg diagonalisation test -------'
    !N_spin_test = 4
    !Sz_choice = 0.0d0 ! integer counted from Sz_max (in a sense that Sz_choice = 1 means eg for N_spin=4, Sz_max=2 Sz=2)
    !call Sz_subspace_choice(N_spin_test, Sz_choice, hash_Sz, Sz_subspace_size)
    ! above we confirmed Sz_choice works
    !write(3,*), "#Heisenberg diagonalisation test: N = 4"
    !call H_XXX_subspace_fill_and_diag_fast(N_spin, J_spin, Sz_choice, Sz_subspace_size, hash_Sz, e, x, m) !here only csr3

    !write(*,*) ' dfeast_scsrev eigenvalues found= ', m
    !do i = 1, m
    !    write(*,*) i, e(i)
    !end do

    !write(*,*) ' dfeast_scsrev eigenvec norm:'
    !do i=1,m
    !    norm = 0.0d0
    !    do j=1, Sz_subspace_size
    !        norm = norm + x(j,i)*(x(j,i))
    !    end do
    !    write(*,*) i, norm
    !   write(1,*) e(i), ',' , Sz_choice, ',', norm
        
    !end do

    !write(*,*) '------- END Heisenberg diagonalisation test -------'

    write(*,*) '------- Start Heisenberg diagonalisation loop -------'

    !here write if describing the situation if the size of the matrix is an integer ;)

    !do k = 1, size(target_sz)
    !Sz_choice = target_sz(k)
    if (mod(N_spin, 2) == 0) then
        ! This is the case for even numbers
        Sz_choice = 0.0d0
    else
        ! This is the case for odd numbers
        Sz_choice = 0.5d0
    endif

    write(*,*) 'Start loop for Sz =', Sz_choice
    write(3,*), "#Heisenberg diagonalisation: N = " , N_spin_char

    call Sz_subspace_choice(N_spin, Sz_choice, hash_Sz, Sz_subspace_size)
    call H_XXX_subspace_fill_and_diag_fast_time_window(N_spin, J_spin, Sz_choice, Sz_subspace_size, hash_Sz, e, x, m)

    write(*,*) ' dfeast_scsrev eigenvalues found= ', m
    write(*,*) ' dfeast_scsrev eigenvec norm:'
    do i=1,m
        norm = 0.0d0
        do j=1, Sz_subspace_size
            norm = norm + x(j,i)*(x(j,i))
            write(4,*) x(j,i)
        end do
        write(2,*) e(i), ',' , Sz_choice, ',', norm
        write(4,*) ''
    end do 


    deallocate(e, x, hash_Sz)

    !end do

    deallocate(target_sz)
    write(*,*) '------- END Heisenberg diagonalisation loop -------'  
    write(*,*) " "
    write(*,*) "Program executed with success"
    finish = omp_get_wtime()
    print '("Final time = ",f," seconds.")',finish-start
    write(3, *) "Final time = ", finish-start ," seconds." 
    close(1) 
    close(2)  
    close(3)
    close(4)
    write(*,*) '------- END Heisenberg Program -------'


end program spin_code